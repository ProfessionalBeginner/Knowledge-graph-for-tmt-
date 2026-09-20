"""
Runs the whole proof end to end and narrates it. One command, ~5 minutes on CPU:

    .venv/bin/python prove.py

Five demos:
  1. the model finds the topic of a question using only its own vectors
  2. the graph walk returns that topic's real facts
  3. editing the graph changes what is retrieved, with no retraining
  4. injecting a fact steers which answer the model prefers -- and editing the graph moves it again
  5. with the graph off, output is bit-identical to plain TMT

Demos 4 and 5 also print what does NOT work: the model's top answer stays wrong, and its free text
is gibberish with the graph on or off. See POC_HANDOFF.md.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("core", "graph", "demos", "tests")]

import json
import random
import subprocess
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)

from main import Model                                          # noqa: E402
from graph_hooks import walk, _safe_normalize                   # noqa: E402
from build_graph_tensors import compute_graph_tensors, fact_sentence  # noqa: E402
from ask import lock_on, generate                               # noqa: E402
from inject_test import fact_memory, question, score            # noqa: E402
from tmt_frozen import embed_text                               # noqa: E402
from demo_no_query import retrieve_no_query                     # noqa: E402

CHECKPOINT = str(ROOT / "checkpoints/base-10m-tracefix_snap2.safetensors")
DIM, LAYERS = 768, 16
QUESTIONS = ["fish is found in ", "a knife is used for ", "where does a cow live? ",
             "tell me about the ocean. it is ", "what is a boat made of? "]
ALPHA, TOP_FACTS = -0.1, 3


def head(n, title):
    print(f"\n\n{'=' * 78}\nDEMO {n}: {title}\n{'=' * 78}")


def label(f):
    return f"{f['subject']} {f['relation']} {f['object']}"


def main():
    print(__doc__.strip().split("\n\n")[0])
    model = Model(dim=DIM, layers=LAYERS, temp=0.75, lr=5e-4)
    model.load(CHECKPOINT)
    model.freeze()

    graph = json.load(open(ROOT / "data/real_graph.json", encoding="utf-8"))
    facts, nodes = graph["facts"], [n["name"] for n in graph["nodes"]]
    raw = mx.load(str(ROOT / "data/real_graph_tensors.safetensors"))
    K, A, F, M, K_ctx = raw["K"], raw["A"], raw["F"], raw["M"], raw["K_ctx"]
    deg = mx.maximum(mx.sum(M, axis=0), 1.0)
    subjects = sorted({f["subject"] for f in facts})
    is_concept = mx.array([1.0 if n in subjects else 0.0 for n in nodes])
    decays = [mx.sigmoid(layer.decay) for layer in model.layers]
    print(f"\ngraph: {len(subjects)} concepts, {len(nodes)} nodes, {len(facts)} facts, "
          f"8 standardized relations. model: 10M params, frozen.")

    # ---------------------------------------------------------------- demo 1 + 2
    head(1, "the model finds the topic itself, then the graph walk reads up on it")
    print("No text search: the question is read byte by byte, and the model's own latent is matched\n"
          "against the graph's node vectors. Nothing is trained in this step.\n")
    locks = []
    for q in QUESTIONS:
        seed, idx, conf = lock_on(model, DIM, LAYERS, q.encode("utf-8"), K_ctx, is_concept, 0.01)
        p = walk(seed, A)
        fscore = _safe_normalize(mx.maximum(M @ (p / deg), 0.0))
        top = mx.argsort(fscore)[::-1][:TOP_FACTS].tolist()
        node = nodes[mx.argmax(seed).item()]
        locks.append((q, node, [facts[i] for i in top]))
        print(f"  {q!r}\n    locked on {node!r} (confidence {conf:.2f}) at {q[max(0, idx - 7):idx + 1]!r}")
        print(f"    graph returned: {[label(facts[i]) for i in top]}")
    ok = sum(1 for q, node, _ in locks if node in q)
    print(f"\n  -> locked on to a word actually in the question: {ok}/{len(QUESTIONS)}")

    # ---------------------------------------------------------------- demo 2b
    head(2, "every concept retrieves its own facts (all 82, not just the examples)")
    hit = precision = 0
    for c in subjects:
        _, _, fs = retrieve_no_query(embed_text(model, DIM, LAYERS, c), K, A, F, M)
        top = [facts[i]["subject"] for i in mx.argsort(fs)[::-1][:3].tolist()]
        hit += c in top
        precision += sum(t == c for t in top) / 3
    print(f"  own fact in the top 3: {hit}/{len(subjects)} concepts")
    print(f"  share of the top 3 that is the concept's own fact: {precision / len(subjects):.0%}")

    # ---------------------------------------------------------------- demo 3
    head(3, "edit the graph and retrieval follows -- no retraining, weights untouched")
    rng = random.Random(0)
    rel_objects = {}
    for f in facts:
        rel_objects.setdefault(f["relation"], set()).add(f["object"])
    edits = []
    for f in rng.sample(facts, len(facts)):
        if len(edits) == 3:
            break
        s, rel, o = f["subject"], f["relation"], f["object"]
        _, _, fs = retrieve_no_query(embed_text(model, DIM, LAYERS, s), K, A, F, M)
        before = [label(facts[i]) for i in mx.argsort(fs)[::-1][:8].tolist()]
        if label(f) not in before or any(e[0] == s for e in edits):
            continue
        new_o = sorted(x for x in rel_objects[rel] if x not in (s, o))[3]
        edits.append((s, rel, o, new_o, before))
    edited = [dict(f) for f in facts]
    for s, rel, o, new_o, _ in edits:
        for f in edited:
            if (f["subject"], f["relation"], f["object"]) == (s, rel, o):
                f["object"] = new_o
    print("  rebuilding graph vectors with the edits (offline, same frozen checkpoint)...")
    K2, F2, A2, M2, _ = compute_graph_tensors(model, DIM, LAYERS, graph["nodes"], edited)
    passed = 0
    for s, rel, o, new_o, before in edits:
        _, _, fs = retrieve_no_query(embed_text(model, DIM, LAYERS, s), K2, A2, F2, M2)
        after = [label(edited[i]) for i in mx.argsort(fs)[::-1][:8].tolist()]
        ok = f"{s} {rel} {new_o}" in after and f"{s} {rel} {o}" not in after
        passed += ok
        print(f"  '{s} {rel} {o}' -> '{s} {rel} {new_o}': {'FOLLOWED' if ok else 'did not follow'}")
        print(f"     retrieved now: {after[:3]}")
    print(f"\n  -> {passed}/{len(edits)} edits followed by retrieval, model never retrained")

    # ---------------------------------------------------------------- demo 4
    head(4, "injecting a fact steers which answer the model prefers")
    print("Same question, same model. The two runs differ ONLY in which fact is held in memory.\n"
          "The number is how much the model prefers the TRUE word over a FALSE one; higher = more.\n")
    rng = random.Random(7)
    follow = total = 0
    for rel in ["AtLocation", "HasA", "UsedFor", "HasProperty", "CapableOf"]:
        idx = [i for i, f in enumerate(facts) if f["relation"] == rel]
        cands = sorted({facts[i]["object"] for i in idx})
        for fi in rng.sample(idx, 3):
            f = facts[fi]
            s, o = f["subject"], f["object"]
            o_false = rng.choice([c for c in cands if c != o])
            q = question(s, rel).encode("utf-8")
            mem_true = fact_memory(model, DIM, LAYERS, fact_sentence(f))
            mem_false = fact_memory(model, DIM, LAYERS, fact_sentence(
                {"subject": s, "relation": rel, "object": o_false}))
            io, iw = cands.index(o), cands.index(o_false)
            m_off = score(model, DIM, LAYERS, None, q, cands)
            m_true = score(model, DIM, LAYERS, None, q, cands, mem=ALPHA * mem_true)
            m_false = score(model, DIM, LAYERS, None, q, cands, mem=ALPHA * mem_false)
            d_off, d_true, d_false = (x[io] - x[iw] for x in (m_off, m_true, m_false))
            good = d_true > d_false
            follow += good
            total += 1
            if total <= 6:
                print(f"  '{question(s, rel)}___'   true={o!r}  false fact says={o_false!r}")
                print(f"     no injection {d_off:+.3f} | true fact injected {d_true:+.3f} | "
                      f"false fact injected {d_false:+.3f}   -> {'follows the fact' if good else 'no'}")
    print(f"\n  -> answer followed the injected fact in {follow}/{total} ({follow / total:.0%}); "
          f"50% would mean injection does nothing")
    print("     (full runs: 80-93% over 5 fresh sets of 56 questions; and 91% (84/92) for the whole\n"
          "      chain -- retrieve from the graph, edit one fact, answer moves. See POC_HANDOFF.md.)")

    # ---------------------------------------------------------------- demo 5
    head(5, "graph off is bit-identical to plain TMT")
    out = subprocess.run([sys.executable, str(ROOT / "tests/test_invisible.py")], capture_output=True, text=True)
    print("  " + "\n  ".join(l for l in out.stdout.strip().split("\n") if l.strip()))

    # ---------------------------------------------------------------- what fails
    head("X", "what does NOT work (run it yourself, don't take our word)")
    print("The model's free text is gibberish with the graph on OR off -- it is 10M params trained\n"
          "on ~0.9MB and forgets after ~3-4 bytes. Injection also costs fluency (~0.5 log-prob per\n"
          "byte), because the model was never trained with anything held in memory.\n")
    for q, node, retrieved in locks[:3]:
        qb = q.encode("utf-8")
        mem = mx.sum(mx.stack([fact_memory(model, DIM, LAYERS, fact_sentence(f))
                               for f in retrieved]), axis=0) * ALPHA
        held = [mem[j] * (1.0 - decays[j]) for j in range(LAYERS)]
        print(f"  {q!r}")
        print(f"    GRAPH OFF: {generate(model, DIM, LAYERS, qb, 44, None, True)!r}")
        print(f"    GRAPH ON:  {generate(model, DIM, LAYERS, qb, 44, held, True)!r}")
    print("\n  Neither is an answer. GRAPH OFF usually loops; GRAPH ON invents misspelled words.")
    print("  The top answer in demo 4 also stays wrong either way (it picks 'house' for almost")
    print("  everything) -- injection shifts preferences, it does not fix the model.")
    print("\n  Fix: train the model WITH injection active so held memory is in-distribution.")
    print("  See 'Next steps' in POC_HANDOFF.md.")


if __name__ == "__main__":
    main()
