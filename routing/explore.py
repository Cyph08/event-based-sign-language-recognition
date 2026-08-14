"""
routing/explore.py
------------------
Broad search for the best achievable accuracy, run unattended.

PROTOCOL (important for the viva): configurations are ranked by **validation**
accuracy. The test score of the validation-winner is what gets reported. Picking
the best *test* number out of many runs is p-hacking and will not survive
scrutiny — the summary therefore prints val-ranked, and flags the val winner.

Realistic ceilings, from the literature:
  SL-Animals   published SOTA ~77-78% (reduced set). 90% is NOT plausible here.
  DVSGesture   published SOTA ~98.8%. 90%+ IS plausible — most headroom is here.

    python explore.py              # run the whole queue
    python explore.py --summary    # ranked table of everything so far
"""

import argparse
import glob
import json
import os
import sys
import time
import traceback

import dataset as D
from mark_model import train_and_eval

RESULTS = os.path.join(D.DATA, "results")
LOG = os.path.join(RESULTS, "explore.log")


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


# ----------------------------------------------------------------------
# the queue: (code, dataset, mode, net, extra kwargs)
# ordered by expected value — biggest headroom first
# ----------------------------------------------------------------------
QUEUE = [
    # 1. DVSGesture with the winning recipe — this is where 90% is reachable
    ("D", "dvs", "full", "deep",        dict(T=16, epochs=40)),
    ("D", "dvs", "full", "deep_ghost",  dict(T=16, epochs=40)),
    ("D", "dvs", "hand", "deep",        dict(T=16, epochs=40)),

    # 2. SL: combine the two architectures that actually won
    ("E", "sl", "full", "deep_ghost",   dict(T=16, epochs=40)),
    ("E", "sl", "full", "deep5",        dict(T=16, epochs=40)),
    ("E", "sl", "full", "deep_drop",    dict(T=16, epochs=40)),
    ("E", "sl", "full", "deep5_ghost",  dict(T=16, epochs=40)),
    ("E", "sl", "full", "deep_ghost_drop", dict(T=16, epochs=40)),

    # 3. more timesteps on the best-so-far architecture
    ("F", "sl", "full", "deep",         dict(T=32, epochs=30)),
    ("F", "dvs", "full", "deep",        dict(T=32, epochs=30)),

    # 4. longer training
    ("G", "sl", "full", "deep",         dict(T=16, epochs=80)),
    ("G", "dvs", "full", "deep",        dict(T=16, epochs=80)),

    # 5. hand arm with the best architecture (keeps the A/B story matched)
    ("H", "sl", "hand", "deep_ghost",   dict(T=16, epochs=40)),
    ("H", "dvs", "hand", "deep_ghost",  dict(T=16, epochs=40)),
]

AUG = dict(augment=True, mix_p=0.3, smooth=0.1)


def tag_for(code, ds, mode, net, T, seed=0, keep_frac=0.5):
    t = f"{code}_{ds}_{mode}_{net}_T{T}_seed{seed}"
    if mode in ("hand", "crop64"):
        t += f"_keep{keep_frac}"
    return t + "_aug"


def run_one(code, ds, mode, net, kw):
    tag = tag_for(code, ds, mode, net, kw.get("T", 16))
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
    rows = []
    for p in glob.glob(os.path.join(RESULTS, "*.json")):
        if os.path.basename(p).startswith(("ensemble", "run_all", "explore")):
            continue
        d = json.load(open(p))
        rows.append((d["dataset"], d["mode"], d["net"], d.get("T"),
                     d.get("augment", False), d["params"],
                     d["best_val_acc"], d["test_acc"]))
    for ds in ("sl", "dvs"):
        sub = [r for r in rows if r[0] == ds]
        if not sub:
            continue
        sub.sort(key=lambda r: -r[6])            # rank by VALIDATION
        log(f"\n===== {ds.upper()} — ranked by VALIDATION (protocol: val picks, test reports) =====")
        log(f"{'mode':7s}{'net':18s}{'T':4s}{'aug':5s}{'params':>9s}{'val':>8s}{'test':>8s}")
        log("-" * 60)
        for i, (_, m, n, t, a, pr, v, te) in enumerate(sub):
            mark = "  <-- val winner" if i == 0 else ""
            log(f"{m:7s}{n:18s}{str(t):4s}{'Y' if a else 'N':5s}{pr:>9,}"
                f"{v:>8.3f}{te:>8.3f}{mark}")


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
        log(f"\nQUEUE COMPLETE — {(time.time()-t0)/60:.0f} min")
        summary()
