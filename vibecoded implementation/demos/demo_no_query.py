"""
PoC: the fix for the "always matches goat" collapse (see RUNLOG.md in archive_madeup/). Instead of a *learned*
W_q that gradient descent can (and did) collapse into ignoring its input, use the frozen model's
own latent directly as the query -- there's nothing trainable in the matching step left to
collapse. Graph traversal (the existing, non-learned PPR walk in graph_hooks.walk()) does all the
actual "find related facts" work; this file only does the query + cosine-match step differently.

No training involved anywhere in this script. Uses real_graph.json (~100 real everyday concepts,
8 ConceptNet relation names) to show retrieval discriminates between different queries, instead
of collapsing to one node regardless of input.
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
from tmt_frozen import embed_text


def retrieve_no_query(z, K, A, F, M, tau=0.01, alpha=0.5, walk_steps=10):
    """Same math as GraphHooks.retrieve(), minus the learned W_q -- z itself is the query."""
    cos = (K @ z) / (mx.linalg.norm(K, axis=1) * mx.linalg.norm(z) + 1e-8)
    seed = mx.softmax(cos / tau)
    p = walk(seed, A, alpha=alpha, steps=walk_steps)
    # Degree-normalize before scoring facts: PPR mass pools on high-degree hubs (shared values
    # like "house"/"kitchen"/"animal" that many concepts point at), so raw M @ p ranks facts about
    # the hub above facts actually about the query. Dividing by node degree removes that bias.
    # tau=0.01: this model's latents are all fairly similar (it only remembers the last ~3-4
    # bytes), so a softer softmax smears the walk's seed over many nodes -- at tau=0.1 only 56/82
    # concepts got their own facts in the top 3; at 0.01 all 82 do.
    deg = mx.maximum(mx.sum(M, axis=0), 1.0)
    fscore = _safe_normalize(mx.maximum(M @ (p / deg), 0.0))
    r = fscore @ F
    return r, seed, fscore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/base-10m-tracefix_snap2.safetensors"))
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--graph", default=str(ROOT / "data/real_graph.json"))
    ap.add_argument("--graph-tensors", default=str(ROOT / "data/real_graph_tensors.safetensors"))
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--queries", default="ocean,fish,knife,bird,fire,kitchen",
                    help="comma-separated words to query with")
    args = ap.parse_args()

    model = Model(dim=args.dim, layers=args.layers, temp=0.75, lr=5e-4)
    model.load(args.checkpoint)
    model.freeze()

    graph = json.load(open(args.graph, encoding="utf-8"))
    node_names = [n["name"] for n in graph["nodes"]]
    fact_labels = [f"{f['subject']} {f['relation']} {f['object']}" for f in graph["facts"]]

    raw = mx.load(args.graph_tensors)
    K, A, F, M = raw["K"], raw["A"], raw["F"], raw["M"]

    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    for q_name in queries:
        z = embed_text(model, args.dim, args.layers, q_name)
        r, seed, fscore = retrieve_no_query(z, K, A, F, M, tau=args.tau)

        top_nodes = mx.argsort(seed)[::-1][:5].tolist()
        top_facts = mx.argsort(fscore)[::-1][:5].tolist()

        own = [f"{f['subject']} {f['relation']} {f['object']}" for f in graph["facts"]
               if f["subject"] == q_name][:5]
        print("=" * 70)
        print(f"QUERY: {q_name!r}   (in graph: {q_name in node_names}; its own facts: {own})")
        print(f"  top-5 matched nodes: {[(node_names[i], round(seed[i].item(), 4)) for i in top_nodes]}")
        print(f"  top-5 scored facts:  {[(fact_labels[i], round(fscore[i].item(), 4)) for i in top_facts]}")


if __name__ == "__main__":
    main()
