"""
routing/explore_dvs_seeds.py
----------------------------
Error bars for DVSGesture. This is the run that makes the DVS numbers citable.

Why: with the augmentation fix, DVS full-frame gave 91.3% and 83.7% on two seeds
(mean 87.5 +- 5.4) while the hand arm gave 86.7/85.6 (86.2 +- 0.8). The full arm's
+-5.4 makes every DVS claim — including the A/B comparison — underpowered. More
seeds is the only fix; a single T=32 run would not have addressed it.

Runs seeds 2,3,4 on both arms with the DVS preset (no flip/rotate).

    python explore_dvs_seeds.py             # run
    python explore_dvs_seeds.py --summary   # mean +- std and the A/B t-test
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
LOG = os.path.join(RESULTS, "explore_dvs_seeds.log")
NET = "deep5_ghost"
STABLE = dict(patience=15, lr=5e-4, epochs=50, T=16)
AUG = dict(augment=True, mix_p=0.3, smooth=0.1)
NEW_SEEDS = (2, 3, 4)


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


# full arm first — it is the one with the variance problem
QUEUE = ([("full", s) for s in NEW_SEEDS] + [("hand", s) for s in NEW_SEEDS])


def run_one(mode, seed):
    tag = f"DX_dvs_{mode}_{NET}_T16_seed{seed}"
    if mode == "hand":
        tag += "_keep0.5"
    tag += "_aug"
    if os.path.exists(os.path.join(RESULTS, f"{tag}.json")):
        log(f"SKIP {tag}")
        return
    log(f"\n{'='*70}\nRUN  {tag}\n{'='*70}")
    t0 = time.time()
    try:
        acc = train_and_eval(code="DX", ds="dvs", mode=mode, net=NET,
                             seed=seed, **AUG, **STABLE)
        log(f"DONE {tag}: test {acc:.3f}  ({(time.time()-t0)/60:.0f} min)")
    except Exception:
        log(f"FAILED {tag}\n{traceback.format_exc()}")


def summary():
    g = {}
    for p in glob.glob(os.path.join(RESULTS, "DX_dvs_*.json")):
        d = json.load(open(p))
        if d["net"] == NET and d.get("T") == 16:
            g.setdefault(d["mode"], []).append(d["test_acc"])
    log("\n=== DVSGesture, deep5_ghost, T=16, fixed augmentation ===")
    for m in ("full", "hand"):
        a = sorted(g.get(m, []))
        if not a:
            continue
        sd = statistics.stdev(a) if len(a) > 1 else 0.0
        log(f"{m:5s} n={len(a)}  {100*statistics.mean(a):5.1f}% +- {100*sd:4.1f}   "
            f"{[round(100*x, 1) for x in a]}")
    f, h = g.get("full", []), g.get("hand", [])
    if len(f) > 1 and len(h) > 1:
        mf, mh = statistics.mean(f), statistics.mean(h)
        se = (statistics.stdev(f) ** 2 / len(f) +
              statistics.stdev(h) ** 2 / len(h)) ** 0.5
        t = (mf - mh) / se
        log(f"\ngap {100*(mf-mh):.1f} pts | SE {100*se:.2f} | t={t:.2f} -> "
            + ("SIGNIFICANT" if abs(t) > 2.4 else "not significant"))
        log(f"(n={len(f)}/{len(h)} — with this many seeds, a real effect smaller "
            f"than ~{100*2.4*se:.1f} pts would go undetected)")


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
        for mode, seed in QUEUE:
            run_one(mode, seed)
        log(f"\nSEEDS COMPLETE — {(time.time()-t0)/60:.0f} min")
        summary()
