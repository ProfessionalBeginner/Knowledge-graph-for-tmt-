"""
PoC check for the "edit the graph, predictions follow, no retraining" part of the claim, using the
no-learned-query retrieval path (demo_no_query.retrieve_no_query).

For each (subject, relation) picked below: query with the subject word, record what retrieval
returns, then swap that fact's object in the graph for a different real node, rebuild the graph
tensors (offline, from the same frozen checkpoint -- no weights change), and query again. Pass =
the edited fact shows up in the retrieved facts and the old one no longer does, with the model
itself untouched.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("core", "graph", "demos", "tests")]

import argparse
import copy
import json
import random

import mlx.core as mx

from main import Model
from build_graph_tensors import compute_graph_tensors
from demo_no_query import retrieve_no_query
from tmt_frozen import embed_text

TOP_K = 10


def top_facts(model, args, facts, K, A, F, M, subject):
    z = embed_text(model, args.dim, args.layers, subject)
    _, _, fscore = retrieve_no_query(z, K, A, F, M, tau=args.tau)
    idx = mx.argsort(fscore)[::-1][:TOP_K].tolist()
    return [(facts[i]["subject"], facts[i]["relation"], facts[i]["object"]) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/base-10m-tracefix_snap2.safetensors"))
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--graph", default=str(ROOT / "data/real_graph.json"))
    ap.add_argument("--graph-tensors", default=str(ROOT / "data/real_graph_tensors.safetensors"))
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--n-edits", type=int, default=5)
    args = ap.parse_args()

    model = Model(dim=args.dim, layers=args.layers, temp=0.75, lr=5e-4)
    model.load(args.checkpoint)
    model.freeze()

    graph = json.load(open(args.graph, encoding="utf-8"))
    nodes, facts = graph["nodes"], graph["facts"]
    raw = mx.load(args.graph_tensors)
    K, A, F, M = raw["K"], raw["A"], raw["F"], raw["M"]

    # Pick edits: a fact that retrieval currently surfaces for its own subject (so there is
    # something to move), swapped to another object that same relation uses elsewhere in the graph
    # (so the edit stays well-formed, e.g. "fish AtLocation ocean" -> "fish AtLocation desert").
    rel_objects = {}
    for f in facts:
        rel_objects.setdefault(f["relation"], set()).add(f["object"])
    existing = {(f["subject"], f["relation"], f["object"]) for f in facts}
    edits, used_subjects = [], set()
    order = list(range(len(facts)))
    random.Random(0).shuffle(order)  # spread edits across relations instead of all IsA
    for fi in order:
        f = facts[fi]
        if len(edits) >= args.n_edits:
            break
        s, rel, o = f["subject"], f["relation"], f["object"]
        if s in used_subjects:
            continue
        before = top_facts(model, args, facts, K, A, F, M, s)
        if (s, rel, o) not in before:
            continue
        candidates = sorted(x for x in rel_objects[rel] if x not in (s, o) and (s, rel, x) not in existing)
        if not candidates:
            continue
        new_o = candidates[(fi * 7) % len(candidates)]
        edits.append((fi, s, rel, o, new_o, before))
        used_subjects.add(s)

    if not edits:
        print("no editable facts found (retrieval never surfaces a subject's own facts)")
        return

    edited_facts = copy.deepcopy(facts)
    for fi, s, rel, o, new_o, _ in edits:
        edited_facts[fi]["object"] = new_o
    print(f"rebuilding graph tensors with {len(edits)} edited facts (model weights untouched)...")
    K2, F2, A2, M2, _ = compute_graph_tensors(model, args.dim, args.layers, nodes, edited_facts)

    passed = 0
    for fi, s, rel, o, new_o, before in edits:
        after = top_facts(model, args, edited_facts, K2, A2, F2, M2, s)
        new_in, old_gone = (s, rel, new_o) in after, (s, rel, o) not in after
        ok = new_in and old_gone
        passed += ok
        print("=" * 70)
        print(f"EDIT: '{s} {rel} {o}'  ->  '{s} {rel} {new_o}'   {'PASS' if ok else 'FAIL'}")
        print(f"  before: {[' '.join(t) for t in before[:5]]}")
        print(f"  after:  {[' '.join(t) for t in after[:5]]}")
        print(f"  new fact retrieved (top {TOP_K}): {new_in}   old fact gone: {old_gone}")
    print("=" * 70)
    print(f"{passed}/{len(edits)} edits followed by retrieval with no retraining")


if __name__ == "__main__":
    main()
