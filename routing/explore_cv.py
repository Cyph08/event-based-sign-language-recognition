"""
routing/explore_cv.py — 4-fold subject-independent cross-validation, SL-Animals.

WHY. Published SL-Animals results report 4-fold cross-validation. Every figure in this
project so far comes from ONE fixed 41/9/9 subject split, so the headline number was not
directly comparable and the report had to caveat it as "protocol differs". A supervisor
review asked for the state-of-the-art comparison to be elaborated; running the published
protocol is the strongest available answer.

DESIGN (see code/make_cv_splits.py). 59 signers partitioned into 4 subject-disjoint folds.
Per fold: test = that fold (~15 users), val = 8 seeded users from the rest, train = the
remainder (~36 users). Training-set size is therefore close to the fixed split's 41 users,
so any difference between the CV figure and the fixed-split figure reflects the PROTOCOL
rather than a smaller training set. Every user is tested exactly once.

REPORTING. mean +- std ACROSS FOLDS, which is the format the published figures use
(e.g. SLAYER 60.9 +- 4.6% full set, 78.0 +- 3.1% reduced set).

HONEST NOTE. The CV figure may come in below the fixed-split 90.2%. Four folds average over
easy and hard signer groups, whereas one fixed split samples a single draw. Whatever it
returns is what gets reported.

    python explore_cv.py --summary
"""
import argparse, glob, json, os, statistics, sys, time, traceback
import dataset as D
from mark_model import train_and_eval

RESULTS = os.path.join(D.DATA, "results")
LOG = os.path.join(RESULTS, "explore_cv.log")
BASE = dict(T=16, patience=20, lr=5e-4, epochs=80, batch=8, augment=True,
            mix_p=0.3, smooth=0.1, net="ghostsew12", ds="sl", mode="full")


class _Tee:
    def __init__(s, p):
        s.f = open(p, "a", buffering=1, encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x)
    def flush(s): s.o.flush(); s.f.flush()


# seed 0 across all folds first: a complete CV estimate lands even if the night is cut short
QUEUE = [(f, s) for s in (0, 1) for f in (0, 1, 2, 3)]


def run(fold, seed):
    tag = f"CV{fold}_sl_full_ghostsew12_T16_seed{seed}_aug"
    if os.path.exists(os.path.join(RESULTS, f"{tag}.json")):
        print(f"SKIP {tag}", flush=True); return
    print(f"\n{'='*70}\nRUN  fold {fold}, seed {seed}   [{time.strftime('%H:%M')}]\n{'='*70}",
          flush=True)
    t0 = time.time()
    try:
        a = train_and_eval(code=f"CV{fold}", seed=seed, cv_fold=fold, **BASE)
        print(f"DONE {tag}: test {a:.3f} ({(time.time()-t0)/60:.0f} min)", flush=True)
    except Exception:
        print(f"FAILED {tag}\n{traceback.format_exc()}", flush=True)


def summary():
    per_fold = {}
    for p in glob.glob(os.path.join(RESULTS, "CV*_sl_full_ghostsew12_*.json")):
        d = json.load(open(p))
        per_fold.setdefault(d.get("cv_fold"), []).append(d["test_acc"])
    if not per_fold:
        print("no CV results yet", flush=True); return

    print("\n=== SL-Animals, 4-fold subject-independent CV (ghostsew12, 247k) ===", flush=True)
    print(f"{'fold':6s}{'n':4s}{'mean':>9s}{'std':>7s}   runs", flush=True)
    means = []
    for f in sorted(k for k in per_fold if k is not None):
        a = sorted(per_fold[f])
        m = statistics.mean(a)
        sd = statistics.stdev(a) if len(a) > 1 else 0.0
        means.append(m)
        print(f"{f:<6}{len(a):<4}{100*m:>8.1f}%{100*sd:>7.1f}   "
              f"{[round(100*x, 1) for x in a]}", flush=True)

    if len(means) > 1:
        m, sd = statistics.mean(means), statistics.stdev(means)
        print(f"\nCROSS-VALIDATED: {100*m:.1f}% +- {100*sd:.1f}  "
              f"(mean +- std across {len(means)} folds)", flush=True)
        print("\ncompare, published 4-fold CV on the same dataset:", flush=True)
        for name, val in (("SLAYER (original, full set)", "60.9 +- 4.6%"),
                          ("SLAYER (independent reimpl.)", "54.3 +- 6.1%"),
                          ("SLAYER (optimised reimpl.)", "66.4 +- 5.8%"),
                          ("SNN-HDC", "74.1%"),
                          ("STBP", "~77%"),
                          ("SLAYER (original, reduced set)", "78.0 +- 3.1%")):
            print(f"    {name:34s}{val}", flush=True)
        print(f"    {'THIS WORK (4-fold CV)':34s}{100*m:.1f} +- {100*sd:.1f}%", flush=True)
        print(f"    {'THIS WORK (fixed split, n=4)':34s}90.2 +- 0.9%", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    sys.stdout = _Tee(LOG)
    if a.summary:
        summary()
    else:
        t0 = time.time()
        for f, s in QUEUE:
            if time.time() - t0 > 30.0 * 3600:
                print("\nTIME BUDGET REACHED", flush=True); break
            run(f, s)
        print(f"\nCV COMPLETE — {(time.time()-t0)/60:.0f} min", flush=True)
        summary()
