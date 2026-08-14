"""
routing/explore2.py
-------------------
Round 2, built on what actually won: deep5_ghost (SL 84.8% test / 88.9% val).

Goals, in priority order:
  1. MULTI-SEED error bars on the winner — the single most important missing piece.
     Published SL-Animals results vary +-5 points; without error bars no ranking claim
     is safe, and the A/B "comparable" claim needs them to be defensible.
  2. Push the winner further: more timesteps, longer training.
  3. Matched A/B: run the HAND arm on the SAME winning architecture, so the
     comparison is like-for-like.
  4. Carry the winner to DVSGesture.

    python explore2.py             # run the queue
    python explore2.py --summary   # ranked table + seed statistics
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
LOG = os.path.join(RESULTS, "explore2.log")
WINNER = "deep5_ghost"
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


# Ordered by value-per-GPU-hour. Measured cost at T=16 is ~15 min/run;
# T=32 measured at 168 min for a SMALLER model, so those go LAST — one T=32 run
# costs more than the entire error-bar set, which is the more important result.
QUEUE = (
    # 1. error bars on the winner (SL, full) — seeds 1..3 (seed 0 already done)
    [("S", "sl", "full", WINNER, dict(T=16, epochs=40, seed=s)) for s in (1, 2, 3)] +
    # 2. matched A/B: hand arm on the SAME architecture, same seeds
    [("S", "sl", "hand", WINNER, dict(T=16, epochs=40, seed=s)) for s in (0, 1, 2, 3)] +
    # 3. winner carried to DVSGesture, both arms
    [("P", "dvs", "full", WINNER, dict(T=16, epochs=40)),
     ("P", "dvs", "hand", WINNER, dict(T=16, epochs=40)),
     # 4. longer training (cheap-ish)
     ("P", "sl", "full", WINNER, dict(T=16, epochs=80)),
     # 5. T=32 LAST — ~3-4 h each, but T=32 gave +5.3 pts on `deep`, so it is the
     #    most likely single lever to push toward 90%. Runs overnight.
     ("P", "sl", "full", WINNER, dict(T=32, epochs=30)),
     ("P", "dvs", "full", WINNER, dict(T=32, epochs=30))]
)


def tag_for(code, ds, mode, net, T, seed, keep_frac=0.5):
    t = f"{code}_{ds}_{mode}_{net}_T{T}_seed{seed}"
    if mode in ("hand", "crop64"):
        t += f"_keep{keep_frac}"
    return t + "_aug"


def run_one(code, ds, mode, net, kw):
    tag = tag_for(code, ds, mode, net, kw.get("T", 16), kw.get("seed", 0))
    path = os.path.join(RESULTS, f"{tag}.json")
    if os.path.exists(path):
        log(f"SKIP {tag} (test {json.load(open(path))['test_acc']:.3f})")
        return
    log(f"\n{'='*70}\nRUN  {tag}\n{'='*70}")
    t0 = time.time()
    try:
        acc = train_and_eval(code=code, ds=ds, mode=mode, net=net, **AUG, **kw)
        log(f"DONE {tag}: test {acc:.3f}  ({(time.time()-t0)/60:.0f} min)")
    except Exception:
        log(f"FAILED {tag}\n{traceback.format_exc()}")


def summary():
    """Ranked table + mean/std across seeds for the winner (the number that matters)."""
    rows = []
    for p in glob.glob(os.path.join(RESULTS, "*.json")):
        if os.path.basename(p).startswith(("ensemble", "run_all", "explore")):
            continue
        d = json.load(open(p))
        rows.append(d)

    for ds in ("sl", "dvs"):
        sub = sorted([r for r in rows if r["dataset"] == ds],
                     key=lambda r: -r["best_val_acc"])
        if not sub:
            continue
        log(f"\n===== {ds.upper()} — ranked by VALIDATION =====")
        log(f"{'mode':6s}{'net':18s}{'T':4s}{'seed':5s}{'params':>9s}{'val':>8s}{'test':>8s}")
        log("-" * 60)
        for i, r in enumerate(sub):
            log(f"{r['mode']:6s}{r['net']:18s}{str(r.get('T')):4s}"
                f"{str(r.get('seed', 0)):5s}{r['params']:>9,}"
                f"{r['best_val_acc']:>8.3f}{r['test_acc']:>8.3f}"
                f"{'  <-- val winner' if i == 0 else ''}")

    # seed statistics: the defensible headline numbers
    log("\n===== MULTI-SEED (mean +- std) — use THESE for claims =====")
    groups = {}
    for r in rows:
        k = (r["dataset"], r["mode"], r["net"], r.get("T"))
        groups.setdefault(k, []).append(r["test_acc"])
    for k, accs in sorted(groups.items()):
        if len(accs) < 2:
            continue
        m, sd = statistics.mean(accs), statistics.stdev(accs)
        log(f"{k[0]:4s}{k[1]:6s}{k[2]:18s}T{k[3]:<4} n={len(accs)}  "
            f"{m:.3f} +- {sd:.3f}   (runs: {', '.join(f'{a:.3f}' for a in sorted(accs))})")


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
        for code, ds, mode, net, kw in QUEUE:
            run_one(code, ds, mode, net, kw)
        log(f"\nQUEUE 2 COMPLETE — {(time.time()-t0)/60:.0f} min")
        summary()
