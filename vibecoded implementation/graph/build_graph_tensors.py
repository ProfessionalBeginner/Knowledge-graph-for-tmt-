"""
Phase 2 step 2: precompute node signposts (K), fact vectors (F), adjacency (A), and
fact-to-node incidence (M) from real_graph.json, using the frozen base TMT checkpoint.

Everything here runs offline (this is exactly the kind of text use the spec allows -- "words
are only used offline, to build the graph and to compute signposts once", RUNLOG.md in
archive_madeup/ / graph_hooks.py docstring). At run time, graph_hooks.py never decodes bytes or does string
search; it only reads these precomputed tensors.

For each node/fact item: reset TMT's recurrent state completely (no memory bleed between
items), run the bytes through the frozen model manually (same benchmark.py-style loop as
everywhere else in this project -- no changes to main.py, TTT stays off, trace-carry still
runs within an item), and take the final per-step latent (x after the last layer, right before
the decoder -- the same vector Stage A hooks into) as that item's vector.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("core", "graph", "demos", "tests")]

import argparse
import json
import mlx.core as mx
from main import Model
from tmt_frozen import embed_text


RELATION_TEMPLATE = {
    "IsA": "{s} is a type of {o}",
    "HasProperty": "{s} is {o}",
    "CapableOf": "{s} can {o}",
    "AtLocation": "{s} is found in {o}",
    "HasA": "{s} has a {o}",
    "UsedFor": "{s} is used for {o}",
    "MadeOf": "{s} is made of {o}",
    "PartOf": "{s} is part of {o}",
    "RelatedTo": "{s} is related to {o}",
}


def fact_sentence(fact):
    template = RELATION_TEMPLATE.get(fact["relation"])
    if template:
        return template.format(s=fact["subject"], o=fact["object"]) + "."
    return f"{fact['subject']} {fact['relation']} {fact['object']}."


# Contexts each node word is embedded in for K_ctx (see compute_graph_tensors). A node vector
# built only from the bare word matches that word at the START of a prompt but not mid-sentence:
# this model remembers ~3-4 bytes, so "knife" read after "a " is a different vector (it matched
# the node "fire"). Averaging over a few ordinary lead-ins makes the node vector context-robust.
K_CONTEXTS = ["", "the ", "a ", "about the ", "is "]


def compute_graph_tensors(model, dim, layers, nodes, facts):
    """Reusable core: also called by test_graph_edit.py, which rebuilds K/F/A/M after editing
    facts in the graph, with no retraining."""
    node_index = {n["name"]: i for i, n in enumerate(nodes)}

    K_rows = [embed_text(model, dim, layers, n["name"]) for n in nodes]
    K = mx.stack(K_rows)

    K_ctx = mx.stack([mx.mean(mx.stack([embed_text(model, dim, layers, c + n["name"])
                                        for c in K_CONTEXTS]), axis=0) for n in nodes])

    F_rows = [embed_text(model, dim, layers, fact_sentence(f)) for f in facts]
    F = mx.stack(F_rows)

    n_nodes, n_facts = len(nodes), len(facts)
    A_counts = [[0.0] * n_nodes for _ in range(n_nodes)]
    M = [[0.0] * n_nodes for _ in range(n_facts)]
    for fi, f in enumerate(facts):
        si = node_index[f["subject"]]
        M[fi][si] = 1.0
        if f["object"] in node_index:
            oi = node_index[f["object"]]
            M[fi][oi] = 1.0
            A_counts[si][oi] += 1.0
            A_counts[oi][si] += 1.0  # undirected: an edge is a shared fact either way

    A_mx = mx.array(A_counts)
    row_sums = mx.sum(A_mx, axis=1, keepdims=True)
    row_sums = mx.where(row_sums == 0, mx.ones_like(row_sums), row_sums)  # isolated nodes: no-op row
    A = A_mx / row_sums
    M = mx.array(M)
    mx.eval(K, K_ctx, F, A, M)
    return K, F, A, M, K_ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/base-10m-tracefix_snap2.safetensors"))
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--graph", default=str(ROOT / "data/real_graph.json"))
    ap.add_argument("--out", default=str(ROOT / "data/real_graph_tensors.safetensors"))
    args = ap.parse_args()

    graph = json.load(open(args.graph, encoding="utf-8"))
    nodes = graph["nodes"]
    facts = graph["facts"]

    model = Model(dim=args.dim, layers=args.layers, temp=0.75, lr=5e-4)
    model.load(args.checkpoint)
    model.freeze()

    print(f"embedding {len(nodes)} node signposts and {len(facts)} fact vectors...")
    K, F, A, M, K_ctx = compute_graph_tensors(model, args.dim, args.layers, nodes, facts)

    mx.save_safetensors(args.out, {"K": K, "F": F, "A": A, "M": M, "K_ctx": K_ctx})
    print(f"saved K{tuple(K.shape)}, F{tuple(F.shape)}, A{tuple(A.shape)}, M{tuple(M.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
