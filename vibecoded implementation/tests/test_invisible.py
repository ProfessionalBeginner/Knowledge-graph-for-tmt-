"""
Phase 2 step 3: invisibility test. Hard rule 3: with the graph disabled, outputs must be
bit-identical to unmodified TMT. Runs the same 1000 fixed bytes through:
  (a) plain frozen TMT (no graph_hooks.py involved at all)
  (b) TMT with GraphHooks installed, three configurations:
      - enabled=True, fresh (zero-init W_o) params -- should match (a) because W_o=0 makes the
        injection exactly zero regardless of the gate value
      - enabled=True, params mutated to random nonzero values -- should now DIFFER from (a),
        proving the hook is actually wired in and doing something when live
      - enabled=False, same mutated nonzero params -- should match (a) again, proving the hard
        bypass (not reliance on the gate) is what makes "graph off" truly invisible even after
        training moves the gate/W_o away from their zero-init values (see RUNLOG.md Phase 0 in archive_madeup/)

Uses the same manual frozen-forward pattern as benchmark.py / RUNLOG's Phase 0 differentiability
test (no calls into Model.__call__'s training path, so no TTT, but state/trace still persists
across the 1000 bytes -- hard rules 5 and 6). No changes to main.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("core", "graph", "demos", "tests")]

import mlx.core as mx
from main import Model
from graph_hooks import GraphHooks
from tmt_frozen import reset_state, step

DIM = 768
LAYERS = 16
N_BYTES = 1000
N_NODES = 10
N_FACTS = 5


def run_sequence(model, byte_seq, hooks=None, K=None, A=None, F=None, M=None):
    reset_state(model, DIM)
    dummies = [mx.zeros((DIM,)) for _ in range(LAYERS)]
    gt = {"K": K, "A": A, "F": F, "M": M} if hooks is not None else None
    outputs = []
    for b in byte_seq:
        logits, stop, _, _ = step(model, DIM, dummies, b, hooks, gt)
        outputs.append((logits, stop))
    return outputs


def outputs_identical(out_a, out_b):
    for (la, sa), (lb, sb) in zip(out_a, out_b):
        if not mx.array_equal(la, lb).item():
            return False
        if not mx.array_equal(sa, sb).item():
            return False
    return True


def main():
    mx.random.seed(0)
    model = Model(dim=DIM, layers=LAYERS, temp=0.75, lr=5e-4)
    model.freeze()

    byte_seq = [mx.random.randint(0, 256, (1,)).item() for _ in range(N_BYTES)]

    K = mx.random.normal((N_NODES, DIM))
    A_raw = mx.abs(mx.random.normal((N_NODES, N_NODES)))
    A = A_raw / mx.sum(A_raw, axis=1, keepdims=True)
    F = mx.random.normal((N_FACTS, DIM))
    M = (mx.random.uniform(shape=(N_FACTS, N_NODES)) > 0.7).astype(mx.float32)

    baseline = run_sequence(model, byte_seq)

    hooks = GraphHooks(dim=DIM)
    hooks.enabled = True
    with_fresh_hooks = run_sequence(model, byte_seq, hooks, K, A, F, M)
    check1 = outputs_identical(baseline, with_fresh_hooks)
    print(f"[1] enabled=True, zero-init W_o  vs baseline -> identical: {check1}  (expect True)")

    hooks.w_o.weight = mx.random.normal((DIM, DIM)) * 0.1
    hooks.w_q.weight = mx.random.normal((DIM, DIM)) * 0.1
    hooks.gate.bias = mx.array([2.0])  # force gate open-ish
    with_trained_hooks = run_sequence(model, byte_seq, hooks, K, A, F, M)
    check2 = outputs_identical(baseline, with_trained_hooks)
    print(f"[2] enabled=True, nonzero (mutated) params vs baseline -> identical: {check2}  (expect False)")

    hooks.enabled = False
    with_disabled_hooks = run_sequence(model, byte_seq, hooks, K, A, F, M)
    check3 = outputs_identical(baseline, with_disabled_hooks)
    print(f"[3] enabled=False, same nonzero params vs baseline -> identical: {check3}  (expect True)")

    passed = check1 and (not check2) and check3
    print(f"\nPhase 2 invisibility test: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    ok = main()
    import sys
    sys.exit(0 if ok else 1)
