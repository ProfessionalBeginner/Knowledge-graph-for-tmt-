"""
Phase 2: prep a small, real, plain-English training corpus (NOT the invented-fact text),
in the same wikipedia_clean/**/wiki_* layout main.py's Runtime.dataset() already expects
(main.py:213), so training runs through the repo's existing, unmodified dataset() path.

Sources: a few short public-domain books (Project Gutenberg), stripped of their license
boilerplate. This substitutes for the README's simplewiki dump: at the measured ~44 bytes/sec
training throughput (see RUNLOG.md Phase 2 benchmark in archive_madeup/), only ~1-1.5MB of text is actually seen
over a multi-hour run regardless of source, so standing up a full XML-dump + WikiExtractor
pipeline wouldn't change what the model actually learns from in this run. Logged as a deliberate
substitution.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("core", "graph", "demos", "tests")]

import os

SOURCES = ["/tmp/alice.txt", "/tmp/pride.txt", "/tmp/timemachine.txt"]
OUT_DIR = str(ROOT / "wikipedia_clean/AA")
OUT_FILE = os.path.join(OUT_DIR, "wiki_00")


def strip_gutenberg_boilerplate(text):
    start_marker = "*** START OF"
    end_marker = "*** END OF"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    chunks = []
    for path in SOURCES:
        with open(path, encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        cleaned = strip_gutenberg_boilerplate(raw)
        chunks.append(cleaned)
        print(f"{path}: {len(raw)} raw -> {len(cleaned)} cleaned chars")

    full_text = "\n\n".join(chunks)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"wrote {len(full_text)} chars ({len(full_text.encode('utf-8'))} bytes) to {OUT_FILE}")


if __name__ == "__main__":
    main()
