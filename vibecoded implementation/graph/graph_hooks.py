"""
Phase 3: native graph retrieval hooks for TMT.

Everything in this file is additive -- it is the *only* place with learned parameters outside
TMT itself (W_q, W_o, the gate). TMT's own recurrence/traces/decoder/embedding/base-training
rule are never touched; this module only reads TMT's public per-step latent (the `z` the caller
already computed) and produces either a modified copy of it (Stage A) or a small vector to be
added into the caller's own `dummy` slot for the next step (Stage B) -- see RUNLOG.md Phase 0
in archive_madeup/ for why writing into `dummy` doesn't disturb TMT's own RTRL trace math.

No text at runtime: everything below operates purely on precomputed tensors (K, A, F, M) and
MLX arrays. Nothing here ever decodes a byte or does string search at run time -- names/words
are only used offline, in build_graph_tensors.py, to make K/F once.
"""
import mlx.core as mx
import mlx.nn as nn


def walk(seed: mx.array, A: mx.array, alpha: float = 0.5, steps: int = 10) -> mx.array:
    """Personalized PageRank power iteration. Its own function so the walk regime can be
    swapped later (entropy-gated steps, relation-guided walks, etc. -- out of scope for now,
    see RUNLOG.md 'ideas for later' in archive_madeup/) without touching GraphHooks itself.

    seed: (n_nodes,) distribution over nodes to restart to each step.
    A: (n_nodes, n_nodes) row-normalized adjacency (row i = outgoing distribution from node i).
    Returns p: (n_nodes,) the walk's stationary-ish distribution after `steps` iterations.
    """
    p = seed
    for _ in range(steps):
        p = alpha * seed + (1.0 - alpha) * (A.T @ p)
    return p


def _safe_normalize(x: mx.array, eps: float = 1e-8) -> mx.array:
    total = mx.sum(x)
    return x / (total + eps)


class GraphHooks(nn.Module):
    """Learned params: W_q (dim x dim), W_o (dim x dim, zero-init), gate (dim -> 1, bias -4).
    TMT itself must be frozen wherever this is used -- GraphHooks does not freeze it for you.
    """

    def __init__(self, dim: int, tau: float = 0.1, alpha: float = 0.5, walk_steps: int = 10,
                 gate_bias_init: float = -1.0):
        """gate_bias_init default changed from the spec's suggested -4.0 to -1.0 after three
        consecutive Phase 4 training runs all collapsed to a single node regardless of input --
        see RUNLOG.md Phase 4 runs 1-3 in archive_madeup/. sigmoid'(-4) = 0.018 vs sigmoid'(-1) = 0.197, roughly
        11x more gradient signal to both the injected effect and the gate's own learning at
        init, directly targeting the diagnosed cold-start bottleneck. Does not affect the
        invisibility guarantee (rule 3): that depends entirely on W_o=0 (g * 0 = 0 for any g),
        not on the gate's specific value -- verified this still holds after the change (see
        RUNLOG.md in archive_madeup/). Not one of the plan's 9 numbered hard rules, just a suggested hyperparameter
        in the math section; treating it as tunable given strong empirical justification.
        """
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.alpha = alpha
        self.walk_steps = walk_steps
        self.gate_bias_init = gate_bias_init

        # hard on/off switch -- NOT the same thing as the gate. The gate is learned and will
        # open with training; this is an explicit bypass for the "graph off" baseline/ablations
        # (Phase 5), independent of what the gate has learned. See RUNLOG.md Phase 0 note in
        # archive_madeup/ on why relying on the gate alone isn't a valid "disabled" state post-training.
        self.enabled = True

        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)
        self.w_o.weight = mx.zeros_like(self.w_o.weight)  # spec: W_o starts at zero

        self.gate = nn.Linear(dim, 1, bias=True)
        self.gate.bias = mx.full((1,), gate_bias_init)  # starts closed (see docstring above)

        # Training-time-only warm-up escape hatch, added after runs 1-4 all converged to the
        # gate staying closed (which starves W_q/W_o of gradient, which keeps retrieval useless,
        # which justifies the gate staying closed -- a self-reinforcing trap). When set to a
        # float, retrieve() uses this FIXED value for the actual injection instead of the
        # learned gate, so W_q/W_o get full undiluted gradient and the gate's own params get
        # none (effectively frozen) -- lets retrieval competence bootstrap before the gate has
        # any say. Set back to None to hand control back to the learned gate. Does not change
        # apply_stage_a's formula or the invisibility guarantee (still entirely governed by
        # W_o=0); this only matters while actively training with it non-None.
        self.force_gate = None

    def retrieve(self, z: mx.array, K: mx.array, A: mx.array, F: mx.array, M: mx.array,
                 node_names=None, fact_labels=None, query_z=None):
        """Runs the retrieval math (steps 1-4 of the plan) and the gate. Does NOT inject
        anything -- callers combine the result differently for Stage A vs Stage B.

        z: (dim,) current latent, used for the gate and (if query_z is None) the query.
        query_z: (dim,) optional override for what gets projected into the search query --
            added after diagnosing that z decays to near-identical vectors across different
            subjects within a few bytes of reading a relation phrase (this base model's own
            recency-dominated memory erodes identity fast at its current training level). Lets
            the caller anchor the query to an earlier, less-decayed snapshot (e.g. right after
            reading the subject word) while still using the live z for injection/gating. See
            RUNLOG.md Phase 4 "query anchor" fix in archive_madeup/.
        K: (n_nodes, dim) node signposts.
        A: (n_nodes, n_nodes) row-normalized adjacency.
        F: (n_facts, dim) fact vectors.
        M: (n_facts, n_nodes) fact-to-node incidence.

        Returns (r, g, log) where r is (dim,), g is a python float scalar in (0,1), and log is
        a dict with the gate value and top-3 nodes/facts for later inspection (Phase 5's
        "average gate value when an answer is due vs during filler" analysis, and the
        "what is it looking at" sanity check).
        """
        q = self.w_q(query_z if query_z is not None else z)        # (dim,)
        cos = (K @ q) / (mx.linalg.norm(K, axis=1) * mx.linalg.norm(q) + 1e-8)  # (n_nodes,)
        seed = mx.softmax(cos / self.tau)                          # (n_nodes,)

        p = walk(seed, A, alpha=self.alpha, steps=self.walk_steps)  # (n_nodes,)

        fscore = _safe_normalize(mx.maximum(M @ p, 0.0))            # (n_facts,)
        r = fscore @ F                                              # (dim,)

        g_learned = mx.sigmoid(self.gate(z))[0]                     # scalar
        g = mx.array(self.force_gate) if self.force_gate is not None else g_learned

        top3_node_idx = mx.argsort(seed)[::-1][:3].tolist()
        top3_fact_idx = mx.argsort(fscore)[::-1][:3].tolist()
        log = {
            "gate": g.item(),
            "gate_learned": g_learned.item(),  # what the gate itself would say, even if forced
            "top3_nodes": [(node_names[i] if node_names else i, round(seed[i].item(), 4))
                            for i in top3_node_idx],
            "top3_facts": [(fact_labels[i] if fact_labels else i, round(fscore[i].item(), 4))
                            for i in top3_fact_idx],
        }
        return r, g, log

    def apply_stage_a(self, z: mx.array, K, A, F, M, node_names=None, fact_labels=None, query_z=None):
        """Stage A: z' = z + g * (W_o . r), meant to be injected only into the decoder input.
        Gradient spans one step only (no BPTT needed) -- see RUNLOG.md Phase 0 in archive_madeup/.

        When self.enabled is False, returns (z, None) unchanged -- a hard bypass independent
        of gate/W_o values, for the graph-off baseline (Phase 5) and ablations.
        """
        if not self.enabled:
            return z, None
        r, g, log = self.retrieve(z, K, A, F, M, node_names, fact_labels, query_z=query_z)
        z_prime = z + g * self.w_o(r)
        return z_prime, log

    def stage_b_injection(self, z: mx.array, K, A, F, M, node_names=None, fact_labels=None, query_z=None):
        """Stage B (not wired in yet -- Stage A must be validated first per the plan). Returns
        just the injection vector g * (W_o . r), meant to be added into the caller's `dummy`
        slot for the *next* step's layer input (main.py:69/87), not into z itself.
        """
        if not self.enabled:
            return mx.zeros_like(z), None
        r, g, log = self.retrieve(z, K, A, F, M, node_names, fact_labels, query_z=query_z)
        return g * self.w_o(r), log
