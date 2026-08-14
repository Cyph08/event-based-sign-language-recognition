"""
routing/explore_v5.py — two explicit targets.

TARGET 1: shrink the SL hand-vs-full gap (currently 7.4 pts at keep_frac 0.7).
  The lever is REPRESENTATION, not more events. `hand` mode zeroes 94% of pixels
  but keeps a 128x128 canvas, so the model spends capacity on empty space.
  `crop64` gives a DENSE 64x64 window that follows the hand: 4x fewer pixels,
  all of them signal — and a genuinely smaller input, which strengthens the
  data-reduction claim rather than weakening it.
  Also sweeps keep_frac 0.85 to complete the accuracy-vs-data curve.

TARGET 2: DVSGesture to ~96% (currently 92.8% with dvsnet).
  Levers, in order of expected value:
    T=24   DVS classes ARE motion, so temporal resolution should pay here more
           than on SL. Untested on the residual architecture.
    width  DVS has more training data (915 vs 779) and tolerated base 24.
    longer 150 epochs — the big models were still improving at 80.

    python explore_v5.py --summary
"""
import argparse, glob, json, os, statistics, sys, time, traceback
import dataset as D
from mark_model import train_and_eval

RESULTS = os.path.join(D.DATA, "results")
LOG = os.path.join(RESULTS, "explore_v5.log")
BASE = dict(patience=20, lr=5e-4, batch=8, augment=True, mix_p=0.3, smooth=0.1)

class _Tee:
    def __init__(s, p):
        s.f = open(p, "a", buffering=1, encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x)
    def flush(s): s.o.flush(); s.f.flush()

# (code, ds, mode, net, T, seed, extra)
QUEUE = [
    # --- TARGET 1: close the SL gap ---
    ("CR", "sl",  "crop64", "ghostsew12", 16, 0, dict(keep_frac=0.7, epochs=80)),
    ("CR", "sl",  "crop64", "ghostsew12", 16, 1, dict(keep_frac=0.7, epochs=80)),
    ("KF", "sl",  "hand",   "ghostsew12", 16, 0, dict(keep_frac=0.85, epochs=80)),
    # --- TARGET 2: DVS to 96 ---
    ("DV", "dvs", "full",   "dvsnet",     24, 0, dict(epochs=80)),
    ("DV", "dvs", "full",   "dvsnet",     16, 1, dict(epochs=80)),
    ("DV", "dvs", "full",   "dvsnet",     24, 1, dict(epochs=80)),
    ("DV", "dvs", "full",   "dvsnet",     24, 0, dict(epochs=150)),   # longer
    # --- TARGET 1 continued: does crop64 transfer to DVS too? ---
    ("CR", "dvs", "crop64", "dvsnet",     16, 0, dict(keep_frac=0.7, epochs=80)),
    ("CR", "sl",  "crop64", "ghostsew12", 16, 2, dict(keep_frac=0.7, epochs=80)),
    # --- Section C (kept, lower priority) ---
    ("C",  "sl",  "full",   "tiny",       16, 0, dict(epochs=80)),
    ("C",  "dvs", "full",   "tiny",       16, 0, dict(epochs=80)),
    ("C",  "sl",  "full",   "smallsew",   16, 0, dict(epochs=80)),
    ("C",  "dvs", "full",   "smallsew",   16, 0, dict(epochs=80)),
]

def run(code, ds, mode, net, T, seed, extra):
    kf = extra.get("keep_frac", 0.5)
    tag = f"{code}_{ds}_{mode}_{net}_T{T}_seed{seed}"
    if mode in ("hand", "crop64"):
        tag += f"_keep{kf}"
    if extra.get("epochs", 80) != 80:
        tag += f"_e{extra['epochs']}"
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
    print(f"\n{'ds':5s}{'arm':8s}{'net':13s}{'T':4s}{'kf':5s}{'n':3s}{'mean':>8s}{'std':>7s}",
          flush=True)
    for k in sorted(g):
        a = sorted(g[k]); sd = statistics.stdev(a) if len(a) > 1 else 0.0
        print(f"{k[0]:5s}{k[1]:8s}{k[2]:13s}{str(k[3]):4s}{str(k[4]):5s}{len(a):<3}"
              f"{100*statistics.mean(a):>7.1f}%{100*sd:>7.1f}", flush=True)
    # the two numbers the targets are about
    for ds in ("sl", "dvs"):
        full = [v for k, v in g.items() if k[0] == ds and k[1] == "full"]
        best_full = max((statistics.mean(v) for v in full), default=0)
        for arm in ("hand", "crop64"):
            sub = [v for k, v in g.items() if k[0] == ds and k[1] == arm]
            if sub:
                b = max(statistics.mean(v) for v in sub)
                print(f"{ds.upper()} best full {100*best_full:.1f}% vs best {arm} "
                      f"{100*b:.1f}%  -> gap {100*(best_full-b):.1f}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    sys.stdout = _Tee(LOG)
    if a.summary:
        summary()
    else:
        for q in QUEUE: run(*q)
        print("\nV5 COMPLETE", flush=True); summary()
