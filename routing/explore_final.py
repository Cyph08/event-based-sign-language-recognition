"""
routing/explore_final.py
------------------------
Verification runs for the headline claims. Priority order, highest value first.

Context: ghostsew12 produced SL-Animals 89.5% and DVSGesture 92.0% — the best
results of the project, and (for SL) above every published figure found. DVS has
two seeds agreeing exactly at 92.0%; SL has ONE seed. Across this project four
single-seed highs later regressed toward the mean, so the SL number must be
verified before it is cited anywhere.

  1. SL ghostsew12 seeds 1,2,3   -> error bar on the headline claim
  2. hand arms, both datasets    -> A/B story on the winning architecture
  3. SL ghostsew8                -> completes the SL depth curve (12 vs 8)

Deliberately dropped: SL ghostsew18. On DVS depth 18 scored 5.3 points BELOW
depth 12, and SL has less training data (779 vs 915 samples), so it is very
unlikely to win and costs ~90 min.

    python explore_final.py             # run
    python explore_final.py --summary   # every ghostsew result, with error bars
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
LOG = os.path.join(RESULTS, "explore_final.log")
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
    # 1. verify the headline SL number
    ("sl",  "full", "ghostsew12", 1),
    ("sl",  "full", "ghostsew12", 2),
    ("sl",  "full", "ghostsew12", 3),
    # 2. matched hand arms on the winning architecture
    ("sl",  "hand", "ghostsew12", 0),
    ("dvs", "hand", "ghostsew12", 0),
    ("sl",  "hand", "ghostsew12", 1),
    ("dvs", "hand", "ghostsew12", 1),
    # 3. one more DVS seed + SL depth check
    ("dvs", "full", "ghostsew12", 2),
    ("sl",  "full", "ghostsew8",  0),
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
    g = {}
    for p in glob.glob(os.path.join(RESULTS, "SEW_*.json")):
        d = json.load(open(p))
        g.setdefault((d["dataset"], d["mode"], d["net"]), []).append(d["test_acc"])
    log(f"\n{'ds':5s}{'arm':6s}{'net':13s}{'n':3s}{'mean':>8s}{'std':>7s}   runs")
    log("-" * 62)
    for k in sorted(g):
        a = sorted(g[k])
        sd = statistics.stdev(a) if len(a) > 1 else 0.0
        log(f"{k[0]:5s}{k[1]:6s}{k[2]:13s}{len(a):<3}{100*statistics.mean(a):>7.1f}%"
            f"{100*sd:>7.1f}   {[round(100*x, 1) for x in a]}")
    for ds in ("sl", "dvs"):
        f = g.get((ds, "full", "ghostsew12")); h = g.get((ds, "hand", "ghostsew12"))
        if f and h and len(f) > 1 and len(h) > 1:
            mf, mh = statistics.mean(f), statistics.mean(h)
            se = (statistics.stdev(f) ** 2 / len(f) +
                  statistics.stdev(h) ** 2 / len(h)) ** 0.5
            t = (mf - mh) / se if se else float("inf")
            log(f"\n{ds.upper()} A/B: full {100*mf:.1f}% vs hand {100*mh:.1f}% | "
                f"gap {100*(mf-mh):.1f} | t={t:.2f} -> "
                + ("SIGNIFICANT" if abs(t) > 2.4 else "not significant"))


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
        log(f"\nFINAL QUEUE COMPLETE — {(time.time()-t0)/60:.0f} min")
        summary()
