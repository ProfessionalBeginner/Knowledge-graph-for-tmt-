"""
Phase 2 step 1: train base TMT at ~10M params on real text (no invented facts), using
main.py's existing, unmodified Runtime/Model/Layer machinery -- this script only orchestrates
an outer loop and calls Runtime.call(), the same public entrypoint Runtime.dataset() itself
calls per byte (main.py:221). No changes to main.py, no new training logic: same forward,
same RTRL trace pass, same AdamW step, same save().

Runs for a fixed wall-clock budget (not main.py's infinite streaming loop) so this process
actually exits and the calling agent gets a completion notification instead of having to poll
an unbounded job. Checkpoints periodically via Runtime.save()'s existing every-500-step cadence,
plus a final save on exit (including on Ctrl-C, matching the README's documented behavior).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("core", "graph", "demos", "tests")]

import time
import glob
import itertools
import os
import signal
import sys

from main import Runtime

DIM = 768
LAYERS = 16
CHECKPOINT = os.environ.get("TRAIN_BASE_CHECKPOINT", str(ROOT / "checkpoints/base-10m.safetensors"))
TARGET_SECONDS = float(os.environ.get("TRAIN_BASE_SECONDS", 8 * 3600))

rt = Runtime(path=CHECKPOINT, threshold=0.35, dim=DIM, layers=LAYERS, temp=0.75, lr=5e-4)
rt.model.load(rt.path)

files = sorted(glob.glob(str(ROOT / "wikipedia_clean/**/wiki_*"), recursive=True))
if not files:
    print("no training files found under wikipedia_clean/", file=sys.stderr)
    sys.exit(1)

print(f"training {CHECKPOINT} (dim={DIM}, layers={LAYERS}) on {files} for up to {TARGET_SECONDS}s", flush=True)

start = time.time()
total_bytes = 0
epoch = 0
stop = False


def handle_sigterm(signum, frame):
    global stop
    stop = True


signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

try:
    while time.time() - start < TARGET_SECONDS and not stop:
        epoch += 1
        for file in files:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            data = text.encode("utf-8")
            n_pairs = len(data) - 1
            for i, (c, n) in enumerate(itertools.pairwise(data)):
                rt.call(c, n, i == n_pairs - 1)
                total_bytes += 1
                if total_bytes % 20000 == 0:
                    elapsed = time.time() - start
                    rate = total_bytes / elapsed if elapsed > 0 else 0.0
                    print(f"[{elapsed:7.0f}s] epoch {epoch} total_bytes={total_bytes} "
                          f"rate={rate:.1f} B/s", flush=True)
                if time.time() - start >= TARGET_SECONDS or stop:
                    break
            if time.time() - start >= TARGET_SECONDS or stop:
                break
finally:
    rt.model.save(rt.path)
    elapsed = time.time() - start
    print(f"stopped: {total_bytes} bytes over {epoch} epoch(s) in {elapsed:.0f}s "
          f"({total_bytes / elapsed if elapsed > 0 else 0:.1f} B/s avg), saved to {CHECKPOINT}",
          flush=True)
