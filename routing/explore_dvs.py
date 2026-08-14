"""
routing/explore_dvs.py
----------------------
DVSGesture-specific tuning. SL-Animals settings are left untouched.

WHY THIS EXISTS — a real bug was found in the shared recipe:
  DVSGesture has 6 DIRECTIONAL classes (right/left hand wave; right/left arm
  clockwise; right/left arm counter-clockwise). The shared augmentation flipped
  images left-right 50% of the time, which turns a "Right hand wave" into a
  "Left hand wave" while KEEPING the old label — actively training the model on
  wrong answers for over half the classes. Rotation similarly blurs cw vs ccw.
  Both are now disabled for DVS via augment.AUG_PRESETS['dvs'].

  SL-Animals is unaffected: a sign performed with the other hand is still the
  same sign, so flipping is a valid augmentation there.

Queue order = cheapest, highest-expected-value first:
  1. same model, FIXED augmentation      -> isolates the bug fix
  2. larger models                        -> DVS has more data than SL, so it can
                                             support more capacity
  3. T=32                                 -> DVS classes are defined by MOTION,
                                             so temporal resolution should matter

    python explore_dvs.py             # run
    python explore_dvs.py --summary   # DVS results, before vs after the fix
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
LOG = os.path.join(RESULTS, "explore_dvs.log")
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


# code "DX" = DVS-specific, fixed augmentation
QUEUE = [
    # 1. the bug fix alone, same architecture as before -> direct comparison
    ("DX", "full", "deep5_ghost", dict(seed=0, T=16)),
    ("DX", "full", "deep5_ghost", dict(seed=1, T=16)),
    # 2. more capacity (DVS has 1342 samples vs SL's 1120, and easier classes)
    ("DX", "full", "deep5",       dict(seed=0, T=16)),
    ("DX", "full", "wide",        dict(seed=0, T=16)),
    ("DX", "full", "deep",        dict(seed=0, T=16)),
    # 3. matched hand arm with the fix (keeps the A/B comparison honest)
    ("DX", "hand", "deep5_ghost", dict(seed=0, T=16)),
    ("DX", "hand", "deep5_ghost", dict(seed=1, T=16)),
    # 4. temporal resolution — expensive, so last
    ("DX", "full", "deep5_ghost", dict(seed=0, T=32, epochs=30)),
]


def run_one(code, mode, net, kw):
    seed, T = kw.get("seed", 0), kw.get("T", 16)
    tag = f"{code}_dvs_{mode}_{net}_T{T}_seed{seed}"
    if mode in ("hand", "crop64"):
        tag += "_keep0.5"
    tag += "_aug"
    if os.path.exists(os.path.join(RESULTS, f"{tag}.json")):
        log(f"SKIP {tag}")
        return
    log(f"\n{'='*70}\nRUN  {tag}   (no flip/rotate — DVS preset)\n{'='*70}")
    t0 = time.time()
    kws = dict(STABLE); kws.update(kw)
    try:
        acc = train_and_eval(code=code, ds="dvs", mode=mode, net=net, **AUG, **kws)
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
        if d["dataset"] == "dvs":
            rows.append(d)
    rows.sort(key=lambda r: -r["test_acc"])
    log(f"\n{'arm':6s}{'net':14s}{'T':4s}{'sd':3s}{'aug-fix':8s}{'params':>9s}{'val':>7s}{'test':>7s}")
    log("-" * 62)
    for r in rows:
        log(f"{r['mode']:6s}{r['net']:14s}{str(r.get('T')):4s}{str(r.get('seed',0)):3s}"
            f"{'YES' if r.get('code') == 'DX' else 'no':8s}{r['params']:>9,}"
            f"{r['best_val_acc']:>7.3f}{r['test_acc']:>7.3f}")

    log("\n--- effect of the augmentation fix (deep5_ghost, T16, full) ---")
    for fixed in (False, True):
        a = [r["test_acc"] for r in rows
             if r["net"] == "deep5_ghost" and r["mode"] == "full"
             and r.get("T") == 16 and (r.get("code") == "DX") == fixed]
        if a:
            m = statistics.mean(a)
            sd = statistics.stdev(a) if len(a) > 1 else 0.0
            log(f"{'flip DISABLED' if fixed else 'flip enabled (buggy)':22s} "
                f"n={len(a)}  {100*m:.1f}% +- {100*sd:.1f}")


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
        for code, mode, net, kw in QUEUE:
            run_one(code, mode, net, kw)
        log(f"\nDVS QUEUE COMPLETE — {(time.time()-t0)/60:.0f} min")
        summary()
