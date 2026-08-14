"""
routing/explore3.py
-------------------
Round 3: STABILISE, then measure. This produces the numbers for the dissertation.

Diagnosis from round 2: `deep5_ghost` swung 66.7-84.8% across seeds (std ~9.4).
Every collapsed run early-stopped at 5-7 min; every good run trained 12-16 min.
So the cause is early stopping firing during a mid-training plateau, before the
cosine LR schedule anneals enough to escape it — not an architecture limit.

Fixes applied here:
  patience 5 -> 15   let the cosine schedule finish before giving up
  lr 1e-3 -> 5e-4    gentler steps, fewer bad basins

Goal: same accuracy, much smaller std, so the A/B comparison has real power.
Round-2 A/B was full 76.8 +- 9.4 vs hand 73.7 +- 9.8 (t=0.45) — a gap of 3.1 pts
that the study was UNDERPOWERED to detect. Lower variance fixes that.

    python explore3.py             # run
    python explore3.py --summary   # stabilised vs original, side by side
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
LOG = os.path.join(RESULTS, "explore3.log")
WINNER = "deep5_ghost"
SEEDS = (0, 1, 2, 3)

# the stabilisation settings under test
STABLE = dict(patience=15, lr=5e-4, epochs=50)
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


# code "ST" = stabilised, so results are distinguishable from round 2
QUEUE = ([("ST", "sl", "full", dict(seed=s)) for s in SEEDS] +
         [("ST", "sl", "hand", dict(seed=s)) for s in SEEDS] +
         [("ST", "dvs", "full", dict(seed=s)) for s in SEEDS[:2]] +
         [("ST", "dvs", "hand", dict(seed=s)) for s in SEEDS[:2]])


def run_one(code, ds, mode, kw):
    seed = kw.get("seed", 0)
    tag = f"{code}_{ds}_{mode}_{WINNER}_T16_seed{seed}"
    if mode in ("hand", "crop64"):
        tag += "_keep0.5"
    tag += "_aug"
    if os.path.exists(os.path.join(RESULTS, f"{tag}.json")):
        log(f"SKIP {tag}")
        return
    log(f"\n{'='*70}\nRUN  {tag}   (patience {STABLE['patience']}, lr {STABLE['lr']})\n{'='*70}")
    t0 = time.time()
    try:
        acc = train_and_eval(code=code, ds=ds, mode=mode, net=WINNER,
                             T=16, **AUG, **STABLE, **kw)
        log(f"DONE {tag}: test {acc:.3f}  ({(time.time()-t0)/60:.0f} min)")
    except Exception:
        log(f"FAILED {tag}\n{traceback.format_exc()}")


def summary():
    """Compare stabilised (ST) against the original round-2 runs (S/P)."""
    groups = {}
    for p in glob.glob(os.path.join(RESULTS, "*.json")):
        b = os.path.basename(p)
        if b.startswith(("ensemble", "run_all", "explore")):
            continue
        d = json.load(open(p))
        if d["net"] != WINNER or d.get("T") != 16:
            continue
        kind = "stabilised" if d.get("code") == "ST" else "original"
        groups.setdefault((d["dataset"], d["mode"], kind), []).append(d["test_acc"])

    log(f"\n{'dataset':8s}{'arm':6s}{'config':12s}{'n':3s}{'mean':>8s}{'std':>7s}   runs")
    log("-" * 70)
    for k in sorted(groups):
        a = sorted(groups[k])
        m = statistics.mean(a)
        sd = statistics.stdev(a) if len(a) > 1 else 0.0
        log(f"{k[0]:8s}{k[1]:6s}{k[2]:12s}{len(a):<3}{100*m:>7.1f}%{100*sd:>7.1f}   "
            f"{[round(100*x, 1) for x in a]}")

    # the A/B test that the dissertation rests on
    for ds in ("sl", "dvs"):
        for kind in ("original", "stabilised"):
            f = groups.get((ds, "full", kind)); h = groups.get((ds, "hand", kind))
            if not f or not h or len(f) < 2 or len(h) < 2:
                continue
            mf, mh = statistics.mean(f), statistics.mean(h)
            se = (statistics.stdev(f) ** 2 / len(f) +
                  statistics.stdev(h) ** 2 / len(h)) ** 0.5
            t = (mf - mh) / se if se else float("inf")
            verdict = ("NOT significant — arms indistinguishable" if abs(t) < 2.4
                       else "SIGNIFICANT difference")
            log(f"\n{ds.upper()} {kind}: full {100*mf:.1f}% vs hand {100*mh:.1f}% "
                f"| gap {100*(mf-mh):.1f} pts | t={t:.2f} -> {verdict}")


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
        for code, ds, mode, kw in QUEUE:
            run_one(code, ds, mode, kw)
        log(f"\nQUEUE 3 COMPLETE — {(time.time()-t0)/60:.0f} min")
        summary()
