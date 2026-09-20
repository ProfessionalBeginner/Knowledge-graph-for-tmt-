"""
Snapshots base-10m.safetensors to checkpoints/base-10m_hN.safetensors every hour while
train_base.py keeps running. Purpose: guarantee graph-on vs graph-off comparisons (and any
"does more training help" comparison) are always against a fixed, named weight snapshot, never
a file that might get overwritten mid-pipeline by the still-running training process.

Safe to copy while train_base.py writes concurrently: Model.save() (main.py:146-148) writes to
a temp file then os.replace()'s it into place atomically, so a concurrent copy always sees either
the fully-old or fully-new file, never a partial one.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("core", "graph", "demos", "tests")]

import os
import shutil
import time

SRC = str(ROOT / "checkpoints/base-10m.safetensors")
DEST_DIR = str(ROOT / "checkpoints")
INTERVAL_SECONDS = 3600

os.makedirs(DEST_DIR, exist_ok=True)
hour = 0
print(f"archiving {SRC} -> {DEST_DIR}/ every {INTERVAL_SECONDS}s", flush=True)

while True:
    time.sleep(INTERVAL_SECONDS)
    hour += 1
    if os.path.exists(SRC):
        dest = os.path.join(DEST_DIR, f"base-10m_h{hour}.safetensors")
        shutil.copy2(SRC, dest)
        print(f"[h{hour}] archived -> {dest}", flush=True)
    else:
        print(f"[h{hour}] {SRC} not found yet, skipping", flush=True)
