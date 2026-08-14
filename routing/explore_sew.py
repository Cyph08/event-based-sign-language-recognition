"""
routing/explore_sew.py
----------------------
Residual (SEW) architectures — the attempt to close the gap to published SOTA.

WHY THIS ARCHITECTURE, from our own measurements:
  base 61.4%  ->  deep 69.0%  ->  deep5 76.6%   depth is the strongest lever
  ghost beat base by +5.3 at FEWER parameters   cheap features help
  wide LOST 11 points at 4x the compute         width is the wrong axis
  ... but a plain spiking stack degrades past ~5 blocks, which is exactly where
  deep5 plateaus.

SEW-ResNet (Fang et al., NeurIPS 2021) removes that ceiling by adding the shortcut
to the OUTPUT SPIKES, enabling 100+ layer SNNs. Combining it with Ghost convolutions
halves the cost per block, so the same budget buys roughly twice the depth —
the direction our data says matters most. The literature search found SEW and
Ghost each well studied but not combined inside a spiking residual block.

Controls included on purpose:
  sew12   = same depth WITHOUT Ghost -> isolates what Ghost contributes
  depth 8/12/18 -> shows whether depth still pays once degradation is removed

    python explore_sew.py             # run
    python explore_sew.py --summary   # SEW vs the current best
"""

import argparse
import glob
import json
import os
import statistics
import sys
import time
import traceback

import dataset as D
from mark_model import train_and_eval

RESULTS = os.path.join(D.DATA, "results")
LOG = os.path.join(RESULTS, "explore_sew.log")
# bigger models need longer schedules; batch 8 keeps VRAM ~2 GB
COMMON = dict(T=16, patience=20, lr=5e-4, epochs=80, batch=8)
AUG = dict(augment=True, mix_p=0.3, smooth=0.1)


class _Tee:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1, encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s); self.f.write(s)

    def flush(self):
        self.stdout.flush(); self.f.flush()


def log(m):
    print(m, flush=True)


QUEUE = [
    # DVSGesture first — that is where the 98% target lives
    ("dvs", "full", "ghostsew12", 0),      # the proposed combination
    ("dvs", "full", "sew12",      0),      # CONTROL: same depth, no Ghost
    ("dvs", "full", "ghostsew18", 0),      # does more depth still pay?
    ("dvs", "full", "ghostsew8",  0),      # is shallower enough?
    ("dvs", "full", "ghostsew12", 1),      # seed 2 of the best-looking one
    # then SL-Animals with the same architecture
    ("sl",  "full", "ghostsew12", 0),
    ("sl",  "full", "ghostsew18", 0),
    # and the matched hand arm, so the A/B story keeps up
    ("dvs", "hand", "ghostsew12", 0),
    ("sl",  "hand", "ghostsew12", 0),
]


def run_one(ds, mode, net, seed):
    tag = f"SEW_{ds}_{mode}_{net}_T16_seed{seed}"
    if mode == "hand":
        tag += "_keep0.5"
    tag += "_aug"
    if os.path.exists(os.path.join(RESULTS, f"{tag}.json")):
        log(f"SKIP {tag}")
        return
    log(f"\n{'='*70}\nRUN  {tag}\n{'='*70}")
    t0 = time.time()
    try:
        acc = train_and_eval(code="SEW", ds=ds, mode=mode, net=net, seed=seed,
                             **AUG, **COMMON)
        log(f"DONE {tag}: test {acc:.3f}  ({(time.time()-t0)/60:.0f} min)")
    except Exception:
        log(f"FAILED {tag}\n{traceback.format_exc()}")


def summary():
    rows = []
    for p in glob.glob(os.path.join(RESULTS, "*.json")):
        b = os.path.basename(p)
        if b.startswith(("ensemble", "run_all", "explore")):
            continue
        d = json.load(open(p))
        if d.get("T") == 16 and d["mode"] in ("full", "hand"):
            rows.append(d)
    for ds in ("dvs", "sl"):
        sub = [r for r in rows if r["dataset"] == ds and r["mode"] == "full"]
        if not sub:
            continue
        sub.sort(key=lambda r: -r["test_acc"])
        log(f"\n=== {ds.upper()} full frame — SEW vs previous best ===")
        log(f"{'net':14s}{'sd':3s}{'params':>10s}{'val':>7s}{'test':>7s}")
        log("-" * 44)
        for r in sub[:12]:
            log(f"{r['net']:14s}{str(r.get('seed',0)):3s}{r['params']:>10,}"
                f"{r['best_val_acc']:>7.3f}{r['test_acc']:>7.3f}")
    # does Ghost help inside a residual block? (the control comparison)
    log("\n=== control: Ghost inside SEW blocks (dvs, depth 12) ===")
    for net in ("sew12", "ghostsew12"):
        a = [r["test_acc"] for r in rows
             if r["net"] == net and r["dataset"] == "dvs" and r["mode"] == "full"]
        if a:
            log(f"{net:12s} n={len(a)}  {100*statistics.mean(a):.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    sys.stdout = _Tee(LOG)
    if a.summary:
        summary()
    else:
        t0 = time.time()
        for ds, mode, net, seed in QUEUE:
            run_one(ds, mode, net, seed)
        log(f"\nSEW QUEUE COMPLETE — {(time.time()-t0)/60:.0f} min")
        summary()
