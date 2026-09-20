"""
Ask the model a question and see what it writes back, with the graph off and on.

Pipeline per question (no text search, no outside model at run time -- only vectors move):
  1. LOCK ON. The model reads the question. At the end of each word of 3+ letters its latent is
     cosine-matched against the context-averaged node vectors (K_ctx); the sharpest match wins,
     restricted to nodes that have facts of their own. Why not just query with the latent at the
     end of the question: this model remembers ~3-4 bytes, so that latent reflects "in " rather
     than "fish", and retrieves junk.
  2. WALK. Personalized PageRank from that node over the graph, degree-normalized, top facts.
  3. INJECT. Those facts' precomputed memories are held in the model's recurrent state through the
     `dummy` slot while it writes the answer (see inject_test.py for why held, and why negative).

Fair warning: this 10M-parameter checkpoint is barely fluent, so both answers read as
near-gibberish -- GRAPH OFF typically loops ("at more at more"), GRAPH ON typically produces
misspelled novel text. Holding facts in memory costs fluency (~0.5 log-prob per byte at the
defaults) because the model was never trained with anything held there. Neither output is an
answer; inject_test.py is the measurement showing the injected facts steer the answer, and this
script is only for seeing the raw output. Try --alpha -0.03 (gentler, weaker steering) or
--alpha 0 (identical to GRAPH OFF).

    .venv/bin/python ask.py "what is a boat made of? "
    .venv/bin/python ask.py --alpha -0.3 --n-bytes 80 "the bee has a "
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("core", "graph", "demos", "tests")]

import argparse
import json

import mlx.core as mx

from main import Model
from graph_hooks import walk, _safe_normalize
from inject_test import fact_memory
from tmt_frozen import reset_state, step

DEFAULT_QUESTIONS = [
    "fish is found in ",
    "a knife is used for ",
    "where does a cow live? ",
    "tell me about the ocean. it is ",
]
MIN_WORD = 3  # only lock on at the end of words this long (skips "a", "is", "of")


def lock_on(model, dim, layers, qb, K_ctx, is_concept, tau):
    """Returns (seed distribution over nodes, byte index locked on, confidence)."""
    reset_state(model, dim)
    zeros = [mx.zeros((dim,)) for _ in range(layers)]
    best = None
    word_len = 0
    for i, b in enumerate(qb):
        _, _, _, z = step(model, dim, zeros, b)
        word_len = word_len + 1 if chr(b).isalpha() else 0
        at_word_end = word_len >= MIN_WORD and (i + 1 == len(qb) or not chr(qb[i + 1]).isalpha())
        if not at_word_end:
            continue
        cos = (K_ctx @ z) / (mx.linalg.norm(K_ctx, axis=1) * mx.linalg.norm(z) + 1e-8)
        seed = mx.softmax(cos / tau) * is_concept
        conf = seed.max().item()
        if best is None or conf > best[2]:
            best = (seed / (mx.sum(seed) + 1e-8), i, conf)
    return best


def generate(model, dim, layers, prompt_bytes, n_bytes, held, greedy):
    reset_state(model, dim)
    active = held if held is not None else [mx.zeros((dim,)) for _ in range(layers)]
    logits = None
    for b in prompt_bytes:
        logits, _, _, _ = step(model, dim, active, b)
    out = bytearray()
    for _ in range(n_bytes):
        b = mx.argmax(logits).item() if greedy else model.sample(logits).item()
        out.append(b)
        logits, _, _, _ = step(model, dim, active, b)
    return out.decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="*", help="e.g. \"fish is found in \" (trailing space matters)")
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/base-10m-tracefix_snap2.safetensors"))
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--graph", default=str(ROOT / "data/real_graph.json"))
    ap.add_argument("--graph-tensors", default=str(ROOT / "data/real_graph_tensors.safetensors"))
    ap.add_argument("--alpha", type=float, default=-0.1, help="injection scale (see inject_test.py)")
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--top-facts", type=int, default=3)
    ap.add_argument("--n-bytes", type=int, default=48)
    ap.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    D, L = args.dim, args.layers

    model = Model(dim=D, layers=L, temp=0.75, lr=5e-4)
    model.load(args.checkpoint)
    model.freeze()

    graph = json.load(open(args.graph, encoding="utf-8"))
    facts, nodes = graph["facts"], [n["name"] for n in graph["nodes"]]
    raw = mx.load(args.graph_tensors)
    A, F, M, K_ctx = raw["A"], raw["F"], raw["M"], raw["K_ctx"]
    deg = mx.maximum(mx.sum(M, axis=0), 1.0)
    subjects = {f["subject"] for f in facts}
    is_concept = mx.array([1.0 if n in subjects else 0.0 for n in nodes])
    decays = [mx.sigmoid(layer.decay) for layer in model.layers]

    for q in (args.questions or DEFAULT_QUESTIONS):
        qb = q.encode("utf-8")
        seed, idx, conf = lock_on(model, D, L, qb, K_ctx, is_concept, args.tau)
        p = walk(seed, A)
        fscore = _safe_normalize(mx.maximum(M @ (p / deg), 0.0))
        top = mx.argsort(fscore)[::-1][:args.top_facts].tolist()
        retrieved = [facts[i] for i in top]

        mem = mx.sum(mx.stack([fact_memory(model, D, L, f["subject"] + " " + f["relation"] + " "
                                           + f["object"] + ".") for f in retrieved]), axis=0)
        mem = mem * args.alpha
        held = [mem[j] * (1.0 - decays[j]) for j in range(L)]

        off = generate(model, D, L, qb, args.n_bytes, None, not args.sample)
        on = generate(model, D, L, qb, args.n_bytes, held, not args.sample)
        print("=" * 78)
        print(f"YOU:         {q!r}")
        print(f"locked on:   {nodes[mx.argmax(seed).item()]!r} "
              f"(at {q[max(0, idx - 7):idx + 1]!r}, confidence {conf:.2f})")
        print(f"graph found: {[f['subject'] + ' ' + f['relation'] + ' ' + f['object'] for f in retrieved]}")
        print(f"GRAPH OFF:   {q}{off!r}")
        print(f"GRAPH ON:    {q}{on!r}")


if __name__ == "__main__":
    main()
