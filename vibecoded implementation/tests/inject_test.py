"""
Does injecting a graph fact change the model's answer toward that fact?

Injection here goes into TMT's recurrent memory, not the decoder input. Each layer's memory is
    state = decay * state + enc(byte) + dummy
which is purely additive, so the memory left behind by reading a fact sentence ("fish is found in
ocean.") is a fixed per-layer vector we can precompute offline. Adding it into `state` later (the
`dummy` slot the architecture already has) is like the model having just read that fact -- but at
run time only vectors move: no text, no string search.

Test: for each fact (s, rel, o) ask the model the question "s <rel phrase> " as multiple choice over
every object that relation takes in the graph, and compare:
  off      -- no injection
  true     -- inject the true fact's memory
  swapped  -- inject a *false* fact (s, rel, o') with o' another candidate
  text     -- the true fact sentence actually prepended as text (not the claim; shows whether the
              model can use a fact it just read at all -- it can't, see below)
  graph    -- the full pipeline: subject latent -> cosine match on K -> PPR walk -> top facts ->
              inject their memories. No oracle, no text.
  graph_edited -- same retrieval, but on a graph where the asked fact was edited to o'

Headline metric: "answer leans toward the injected fact" -- per question, compare the model's
preference for o over o' (log-prob margin) between two runs that differ ONLY in which fact is in
memory. 50% means injection content doesn't matter; well above 50% means the answer follows it.
(Top-1 accuracy is reported too but barely moves: this 10M model almost always picks "house".)

Why `hold` mode and a negative alpha (defaults: hold, -0.1, top-3 facts):
  - Every memory dimension in this checkpoint has a half-life of at most ~3-4 bytes (max decay
    0.82). A fact injected once -- or read as text before the question -- is gone before the
    answer starts, which is why `text` never beats chance. `hold` keeps it in memory while the
    model answers.
  - The memory of having just read "... ocean." makes the model LESS likely to write "ocean" next
    (it has just finished that word). With alpha > 0 the answer leans *away* from the injected fact
    (~27-31%); with alpha < 0 it leans toward it. The sign is the one-number stand-in for the
    learned output projection (W_o) the original design has at exactly this point. It was chosen
    on seeds 0-1 and checked on fresh question sets (seeds 2-6).
  - Too much injected memory wrecks the model (log-prob per byte drops by 5-25); -0.1 with the top
    3 facts keeps the damage small. Watch the "mean per-byte log-prob change" line.

For more details, see RUNLOG.md (archived in archive_madeup/).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("core", "graph", "demos", "tests")]

import argparse
import json
import random

import mlx.core as mx

from main import Model
from build_graph_tensors import RELATION_TEMPLATE, fact_sentence
from demo_no_query import retrieve_no_query
from tmt_frozen import reset_state, step, snapshot_state, restore_state


def run_bytes(model, dim, layers, data):
    dummies = [mx.zeros((dim,)) for _ in range(layers)]
    logits = z = None
    for b in data:
        logits, _, _, z = step(model, dim, dummies, b)
    return logits, z


def fact_memory(model, dim, layers, text):
    """Per-layer recurrent state after reading `text` from a clean state: (layers, dim)."""
    reset_state(model, dim)
    run_bytes(model, dim, layers, text.encode("utf-8"))
    return mx.stack([layer.states for layer in model.layers])


def inject(model, mem):
    for j, layer in enumerate(model.layers):
        layer.states = layer.states + mem[j]


def question(s, rel):
    return RELATION_TEMPLATE[rel].split("{o}")[0].format(s=s)


HOLD = True  # set in main() from --mode


def score(model, dim, layers, prefix_bytes, q_bytes, cands, mem=None, inject_after=None):
    """Log-prob of each candidate after [prefix text] + question, with optional memory injection
    right after byte index `inject_after` of the question (None = before the question).

    Two injection modes (--mode):
      once -- add the memory to the state one time; it then fades with the model's own decay
      hold -- keep the memory in the state on every following step, including while the answer is
              scored, by feeding mem * (1 - decay) through each layer's `dummy` input every step
              (the steady state of s = d*s + c is c/(1-d), so the state carries a constant +mem)
    """
    reset_state(model, dim)
    if prefix_bytes:
        run_bytes(model, dim, layers, prefix_bytes)
    zero = [mx.zeros((dim,)) for _ in range(layers)]
    held = zero
    if mem is not None and HOLD:
        held = [mem[j] * (1.0 - mx.sigmoid(model.layers[j].decay)) for j in range(layers)]
    active = zero
    if mem is not None and inject_after is None:
        inject(model, mem)
        active = held
    logits = None
    for i, b in enumerate(q_bytes):
        logits, _, _, _ = step(model, dim, active, b)
        if mem is not None and inject_after == i:
            inject(model, mem)
            active = held
    snap = snapshot_state(model)
    out = []
    for c in cands:
        restore_state(model, snap)
        cb = c.encode("utf-8")
        lp = mx.log(mx.softmax(logits)[cb[0]] + 1e-12)
        cur = logits
        for i in range(len(cb) - 1):
            cur, _, _, _ = step(model, dim, active, cb[i])
            lp = lp + mx.log(mx.softmax(cur)[cb[i + 1]] + 1e-12)
        # length-normalize so short words don't win just for being short
        out.append(lp.item() / len(cb))
        mx.eval([layer.states for layer in model.layers])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/base-10m-tracefix_snap2.safetensors"))
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--graph", default=str(ROOT / "data/real_graph.json"))
    ap.add_argument("--graph-tensors", default=str(ROOT / "data/real_graph_tensors.safetensors"))
    ap.add_argument("--relations", default="AtLocation,HasProperty,IsA,MadeOf,UsedFor,CapableOf,HasA")
    ap.add_argument("--max-per-rel", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=-0.1,
                    help="scale on injected memory (negative: see module docstring)")
    ap.add_argument("--top-facts", type=int, default=3, help="facts injected in the graph condition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concept", default=None,
                    help="run only this concept's questions, e.g. --concept fish (fast: a few seconds)")
    ap.add_argument("--show", type=int, default=8, help="print this many example rows")
    ap.add_argument("--gpu", action="store_true",
                    help="use the GPU (MLX CUDA backend crashes with illegal memory access on long runs)")
    ap.add_argument("--mode", choices=["once", "hold"], default="hold")
    args = ap.parse_args()
    global HOLD
    HOLD = args.mode == "hold"
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    rng = random.Random(args.seed)

    model = Model(dim=args.dim, layers=args.layers, temp=0.75, lr=5e-4)
    model.load(args.checkpoint)
    model.freeze()
    D, L = args.dim, args.layers

    graph = json.load(open(args.graph, encoding="utf-8"))
    facts = graph["facts"]
    raw = mx.load(args.graph_tensors)
    K, A, F, M = raw["K"], raw["A"], raw["F"], raw["M"]

    print(f"precomputing memory for {len(facts)} facts...")
    mems = mx.stack([fact_memory(model, D, L, fact_sentence(f)) for f in facts])  # (n_facts, L, D)
    mx.eval(mems)

    rels = [r for r in args.relations.split(",") if any(f["relation"] == r for f in facts)]
    conds = ["off", "true", "swapped", "text", "graph", "graph_edited"]
    hit = {c: 0 for c in conds}
    follow = {"true_vs_swapped": 0, "text_vs_off": 0, "graph_vs_off": 0, "graph_vs_edited": 0}
    n_retrieved = 0
    lift = {"true": 0.0, "swapped": 0.0, "text": 0.0, "graph": 0.0}
    n = 0
    examples = []

    for rel in rels:
        idx = [i for i, f in enumerate(facts) if f["relation"] == rel]
        cands = sorted({facts[i]["object"] for i in idx})
        if len(cands) < 3:
            continue
        rng.shuffle(idx)
        for fi in idx[:args.max_per_rel]:
            f = facts[fi]
            s, o = f["subject"], f["object"]
            if args.concept and s != args.concept:
                continue
            wrong = [c for c in cands if c != o]
            o_false = rng.choice(wrong)
            false_mem = fact_memory(model, D, L, fact_sentence({"subject": s, "relation": rel, "object": o_false}))
            q = question(s, rel).encode("utf-8")

            # graph condition: query with the model's own latent after reading the subject word
            reset_state(model, D)
            _, z_subj = run_bytes(model, D, L, s.encode("utf-8"))
            _, _, fscore = retrieve_no_query(z_subj, K, A, F, M)
            top = mx.argsort(fscore)[::-1][:args.top_facts]
            graph_mem = mx.sum(mems[top], axis=0)
            # same retrieval, but on a graph where this one fact was edited to the false object:
            # the retrieved slot for fact fi now holds the edited fact's memory
            top_list = top.tolist()
            fact_retrieved = fi in top_list
            edited_mem = graph_mem - mems[fi] + false_mem if fact_retrieved else graph_mem
            retrieved = [f"{facts[i]['subject']} {facts[i]['relation']} {facts[i]['object']}"
                         for i in top.tolist()]

            a = args.alpha
            res = {
                "off": score(model, D, L, None, q, cands),
                "true": score(model, D, L, None, q, cands, mem=a * mems[fi]),
                "swapped": score(model, D, L, None, q, cands, mem=a * false_mem),
                "text": score(model, D, L, (fact_sentence(f) + " ").encode("utf-8"), q, cands),
                "graph": score(model, D, L, None, q, cands, mem=a * graph_mem,
                               inject_after=len(s.encode("utf-8")) - 1),
                "graph_edited": score(model, D, L, None, q, cands, mem=a * edited_mem,
                                      inject_after=len(s.encode("utf-8")) - 1),
            }
            io, iw = cands.index(o), cands.index(o_false)
            margin = {c: res[c][io] - res[c][iw] for c in conds}  # >0 = prefers the true object
            picks = {c: cands[max(range(len(cands)), key=lambda k: res[c][k])] for c in conds}
            for c in conds:
                hit[c] += picks[c] == o
            # paired: does the answer lean toward whichever object the injected fact says?
            follow["true_vs_swapped"] += margin["true"] > margin["swapped"]
            follow["text_vs_off"] += margin["text"] > margin["off"]
            follow["graph_vs_off"] += margin["graph"] > margin["off"]
            n_retrieved += fact_retrieved
            follow["graph_vs_edited"] += fact_retrieved and margin["graph"] > margin["graph_edited"]
            lift["true"] += res["true"][io] - res["off"][io]
            lift["swapped"] += res["swapped"][iw] - res["off"][iw]
            lift["text"] += res["text"][io] - res["off"][io]
            lift["graph"] += res["graph"][io] - res["off"][io]
            n += 1
            if len(examples) < args.show:
                examples.append((s, rel, o, o_false, picks, margin, retrieved))

    if n == 0:
        print("no questions matched (check --concept spelling)")
        return
    print("=" * 78)
    for s, rel, o, o_false, picks, margin, retrieved in examples:
        print(f"Q: '{question(s, rel)}___'  truth={o!r}  false fact says={o_false!r}")
        print("   top pick:        " + "  ".join(f"{c}->{picks[c]}" for c in conds))
        print("   truth-vs-false:  " + "  ".join(f"{c}={margin[c]:+.3f}" for c in conds))
        print(f"   graph retrieved: {retrieved}")
    print("=" * 78)
    print(f"{n} questions, {len(rels)} relations, mode={args.mode}, alpha={args.alpha}, top_facts={args.top_facts}")
    print("top-1 accuracy (multiple choice): " + "  ".join(f"{c}={hit[c] / n:.1%}" for c in conds))
    print("answer leans toward the injected fact (50% = injection does nothing):")
    print(f"  true fact vs false fact injected : {follow['true_vs_swapped'] / n:.1%}")
    print(f"  text-read fact vs nothing        : {follow['text_vs_off'] / n:.1%}")
    print(f"  graph-retrieved facts vs nothing : {follow['graph_vs_off'] / n:.1%}")
    print(f"  FULL PIPELINE, true graph vs graph with that fact edited: "
          f"{follow['graph_vs_edited'] / max(n_retrieved, 1):.1%}  "
          f"(over the {n_retrieved}/{n} questions where retrieval returned the asked fact)")
    print("mean per-byte log-prob change of the injected fact's object vs no injection:")
    print("  " + "  ".join(f"{c}={lift[c] / n:+.4f}" for c in lift))


if __name__ == "__main__":
    main()
