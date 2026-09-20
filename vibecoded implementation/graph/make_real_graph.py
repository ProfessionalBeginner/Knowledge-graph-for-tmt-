"""
Builds real_graph.json from real_graph_table.json: 82 real everyday concepts, every one filling
the same fixed set of standardized slots (a Haiku model filled the table from general knowledge).

Slots (relation names are ConceptNet's):
    IsA, HasProperty, AtLocation, HasA, RelatedTo       -- every concept
    UsedFor / CapableOf                                 -- every concept, one shared slot ("Does"):
                                                           CapableOf for animals/plants/weather/
                                                           nature, UsedFor for everything else
    MadeOf                                              -- objects and foods only

Every slot value becomes a node too, so concepts that share a value ("fish"/"boat" -> ocean,
"knife"/"fork" -> kitchen) are connected through it -- that's what the graph walk travels along.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("core", "graph", "demos", "tests")]

import json

# Hand corrections to the generated table (wrong or filler values).
OVERRIDES = {
    ("ocean", "AtLocation"): "world",
    ("desert", "AtLocation"): "world",
    ("apple", "RelatedTo"): "tree",
    ("horse", "RelatedTo"): "barn",
    ("bed", "RelatedTo"): "pillow",
    ("egg", "MadeOf"): "protein",
    ("meat", "MadeOf"): "protein",
    ("apple", "MadeOf"): "fiber",
    ("banana", "MadeOf"): "fiber",
    ("rice", "MadeOf"): "grain",
}

CAPABLE_TYPES = {"animal", "plant", "weather"}
CAPABLE_CONCEPTS = {"fire", "sun", "moon", "water", "ice", "wood", "metal", "stone", "sand",
                    "river", "ocean", "cloud", "rain", "snow"}
SLOT_ORDER = ["IsA", "HasProperty", "AtLocation", "Does", "HasA", "RelatedTo", "MadeOf"]


def main():
    table = json.load(open(ROOT / "data/real_graph_table.json", encoding="utf-8"))
    nodes, node_type, facts = [], {}, []

    for concept, slots in table.items():
        node_type[concept] = slots["IsA"]
        nodes.append({"name": concept, "type": slots["IsA"]})

    for concept, slots in table.items():
        for slot in SLOT_ORDER:
            if slot not in slots:
                continue
            value = OVERRIDES.get((concept, slot), slots[slot])
            if slot == "Does":
                capable = slots["IsA"] in CAPABLE_TYPES or concept in CAPABLE_CONCEPTS
                relation = "CapableOf" if capable else "UsedFor"
            else:
                relation = slot
            if value not in node_type:
                node_type[value] = "value"
                nodes.append({"name": value, "type": "value"})
            facts.append({"subject": concept, "relation": relation, "object": value})

    with open(ROOT / "data/real_graph.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "facts": facts}, f, indent=2)
    print(f"{len(table)} concepts, {len(nodes)} nodes, {len(facts)} facts -> real_graph.json")


if __name__ == "__main__":
    main()
