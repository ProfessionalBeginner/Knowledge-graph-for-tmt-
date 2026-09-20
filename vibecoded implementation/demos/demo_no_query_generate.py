"""
PoC: same no-learned-query retrieval as demo_no_query.py, but actually injected into generation
so the effect is visible in text, not just in retrieval scores. No training anywhere -- r is
added directly with a small fixed scale (no learned W_o/gate either), since the whole point
tonight is showing the *idea* works, not tuning it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("core", "graph", "demos", "tests")]

import argparse

import mlx.core as mx

from main import Model
from demo_no_query import retrieve_no_query
from tmt_frozen import reset_state, step


def generate(model, dim, layers, prompt_bytes, gt, alpha, n_bytes, temp):
    old_temp = model.temp
    model.temp = temp
    reset_state(model, dim)
    dummies = [mx.zeros((dim,)) for _ in range(layers)]
    logits = None
    for b in prompt_bytes:
        _, _, _, z = step(model, dim, dummies, b)
        if gt is not None:
            r, _, _ = retrieve_no_query(z, gt["K"], gt["A"], gt["F"], gt["M"])
            z = z + alpha * r
        logits, _ = model.decoder(z)
    out = bytearray()
    b = model.sample(logits).item()
    for _ in range(n_bytes):
        out.append(b)
        _, _, _, z = step(model, dim, dummies, b)
        if gt is not None:
            r, _, _ = retrieve_no_query(z, gt["K"], gt["A"], gt["F"], gt["M"])
            z = z + alpha * r
        logits, _ = model.decoder(z)
        b = model.sample(logits).item()
    model.temp = old_temp
    return out.decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/base-10m-tracefix_snap2.safetensors"))
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--graph-tensors", default=str(ROOT / "data/real_graph_tensors.safetensors"))
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--n-bytes", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.5)
    args = ap.parse_args()

    model = Model(dim=args.dim, layers=args.layers, temp=0.75, lr=5e-4)
    model.load(args.checkpoint)
    model.freeze()

    raw = mx.load(args.graph_tensors)
    gt = {"K": raw["K"], "A": raw["A"], "F": raw["F"], "M": raw["M"]}

    for subject in ["ocean", "metal", "kitchen"]:
        prompt = f"Tell me about {subject}. It "
        prompt_bytes = prompt.encode("utf-8")

        # measure injection/z ratio at this prompt to pick a safe alpha (same lesson as
        # demo_cat.py's garbage-output finding in archive_madeup/ -- an unscaled raw r can dominate z easily)
        reset_state(model, args.dim)
        dummies = [mx.zeros((args.dim,)) for _ in range(args.layers)]
        z = None
        for b in prompt_bytes:
            _, _, _, z = step(model, args.dim, dummies, b)
        r, _, _ = retrieve_no_query(z, gt["K"], gt["A"], gt["F"], gt["M"])
        ratio = (mx.linalg.norm(r) / (mx.linalg.norm(z) + 1e-8)).item()
        safe_alpha = min(args.alpha, 0.5 / ratio) if ratio > 0 else args.alpha

        off_text = generate(model, args.dim, args.layers, prompt_bytes, None, 0.0, args.n_bytes, args.temp)
        on_text = generate(model, args.dim, args.layers, prompt_bytes, gt, safe_alpha, args.n_bytes, args.temp)
        print("=" * 70)
        print(f"SUBJECT: {subject!r}  (alpha capped to {safe_alpha:.4f}, raw ratio={ratio:.2f})")
        print(f"  OFF -> {prompt}{off_text!r}")
        print(f"  ON  -> {prompt}{on_text!r}")


if __name__ == "__main__":
    main()
