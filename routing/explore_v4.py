"""
routing/explore_v4.py — three questions, one queue.

1. IS THE SL HAND GAP A BUG?  Pipeline audit found none (retention SL 37.7% vs
   DVS 41.3%, no mask failures, cache keyed correctly, no uint8 saturation,
   sparsity matched). Remaining hypothesis: the mask is too aggressive for
   FINE-GRAINED signs specifically. Test = feed it more events (keep_frac 0.7).
   If accuracy jumps, the mask discards signal; if flat, the 12.4-pt gap is real.

2. DVS-SPECIFIC MODEL. `ghostsew12` was tuned on SL. DVS gestures are large-
   amplitude MOTION with more training data (915 vs 779), so: wider, slightly
   shallower (`dvsnet`), and higher T — temporal resolution should matter more
   where the class IS the motion.

3. SECTION C, PROPERLY. The old C ran at T=8 with no augmentation and scored
   53%/69%. Rebuilt using what the evidence says works (depth + Ghost):
   tiny 38k, smallsew 108k — same recipe as the big models, so the comparison
   is finally like-for-like.

    python explore_v4.py --summary
"""
import argparse, glob, json, os, statistics, sys, time, traceback
import dataset as D
from mark_model import train_and_eval

RESULTS = os.path.join(D.DATA, "results")
LOG = os.path.join(RESULTS, "explore_v4.log")
BASE = dict(patience=20, lr=5e-4, epochs=80, batch=8, augment=True,
            mix_p=0.3, smooth=0.1)


class _Tee:
    def __init__(s, p):
        s.f = open(p, "a", buffering=1, encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x)
    def flush(s): s.o.flush(); s.f.flush()


# (code, ds, mode, net, T, seed, extra)
QUEUE = [
    # 1. is the SL hand mask too aggressive? keep_frac 0.7 vs the 0.5 already run
    ("KF", "sl",  "hand", "ghostsew12", 16, 0, dict(keep_frac=0.7)),
    ("KF", "sl",  "hand", "ghostsew12", 16, 1, dict(keep_frac=0.7)),
    # 2. DVS-specific model, and the untested temporal lever
    ("DV", "dvs", "full", "dvsnet",     16, 0, {}),
    ("DV", "dvs", "full", "ghostsew12", 24, 0, {}),
    ("DV", "dvs", "full", "dvsnet",     24, 0, {}),
    ("DV", "dvs", "full", "dvsnet",     16, 1, {}),
    # 3. Section C rebuilt with the modern recipe, both datasets
    ("C",  "sl",  "full", "tiny",       16, 0, {}),
    ("C",  "sl",  "full", "smallsew",   16, 0, {}),
    ("C",  "dvs", "full", "tiny",       16, 0, {}),
    ("C",  "dvs", "full", "smallsew",   16, 0, {}),
    ("C",  "sl",  "full", "tiny",       16, 1, {}),
    ("C",  "dvs", "full", "tiny",       16, 1, {}),
]


def run(code, ds, mode, net, T, seed, extra):
    kf = extra.get("keep_frac", 0.5)
    tag = f"{code}_{ds}_{mode}_{net}_T{T}_seed{seed}"
    if mode == "hand":
        tag += f"_keep{kf}"
    tag += "_aug"
    if os.path.exists(os.path.join(RESULTS, f"{tag}.json")):
        print(f"SKIP {tag}", flush=True); return
    print(f"\n{'='*70}\nRUN  {tag}\n{'='*70}", flush=True)
    t0 = time.time()
    try:
        a = train_and_eval(code=code, ds=ds, mode=mode, net=net, T=T, seed=seed,
                           **BASE, **extra)
        print(f"DONE {tag}: test {a:.3f} ({(time.time()-t0)/60:.0f} min)", flush=True)
    except Exception:
        print(f"FAILED {tag}\n{traceback.format_exc()}", flush=True)


def summary():
    g = {}
    for p in glob.glob(os.path.join(RESULTS, "*.json")):
        if os.path.basename(p).startswith(("ensemble", "run_all", "explore", "hand_seeds")):
            continue
        d = json.load(open(p))
        g.setdefault((d["dataset"], d["mode"], d["net"], d.get("T"),
                      d.get("keep_frac", 0.5)), []).append(d["test_acc"])
    print(f"\n{'ds':5s}{'arm':6s}{'net':13s}{'T':4s}{'kf':5s}{'n':3s}"
          f"{'mean':>8s}{'std':>7s}{'params':>9s}", flush=True)
    for k in sorted(g):
        a = sorted(g[k]); sd = statistics.stdev(a) if len(a) > 1 else 0.0
        print(f"{k[0]:5s}{k[1]:6s}{k[2]:13s}{str(k[3]):4s}{str(k[4]):5s}{len(a):<3}"
              f"{100*statistics.mean(a):>7.1f}%{100*sd:>7.1f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    sys.stdout = _Tee(LOG)
    if a.summary:
        summary()
    else:
        for q in QUEUE: run(*q)
        print("\nV4 COMPLETE", flush=True); summary()
