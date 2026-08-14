"""
routing/explore_v6.py — overnight queue, ~12 h, ordered by expected value.

THREE THINGS, INTERLEAVED so each gets done even if the night is cut short:

  T1  close the SL hand gap (7.4 pts at keep_frac 0.7)
      -> crop64: DENSE 64x64 window following the hand. Same events, 4x fewer
         pixels. `hand` mode wastes capacity on a 94%-empty 128x128 canvas.

  T2  DVSGesture toward 96% (92.8% now, dvsnet)
      -> T=24 (DVS classes ARE motion), then a 150-epoch run.

  C   Section C done properly. The old C (53%/69%) ran at T=8 with no
      augmentation and was never comparable. Rebuilt with the same recipe as
      the big models: tiny 38k, smallsew 108k.
      Plus the extreme point: tiny 38k ON crop64 64x64 — smallest model AND
      smallest input, the strongest efficiency claim available.

Small models and 64x64 inputs are cheap (~20-30 min), so C costs little.

    python explore_v6.py --summary
"""
import argparse, glob, json, os, statistics, sys, time, traceback
import dataset as D
from mark_model import train_and_eval

RESULTS = os.path.join(D.DATA, "results")
LOG = os.path.join(RESULTS, "explore_v6.log")
BASE = dict(patience=20, lr=5e-4, batch=8, augment=True, mix_p=0.3, smooth=0.1)

class _Tee:
    def __init__(s, p):
        s.f = open(p, "a", buffering=1, encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x)
    def flush(s): s.o.flush(); s.f.flush()

# (code, ds, mode, net, T, seed, extra)  — ordered so each goal advances early
QUEUE = [
    # T1: does a dense crop beat a sparse mask?  (cheap: 64x64)
    ("CR", "sl",  "crop64", "ghostsew12", 16, 0, dict(keep_frac=0.7)),
    ("CR", "sl",  "crop64", "ghostsew12", 16, 1, dict(keep_frac=0.7)),
    # C: the two rebuilt small models, SL first (cheap)
    ("C",  "sl",  "full",   "tiny",       16, 0, {}),
    ("C",  "sl",  "full",   "smallsew",   16, 0, {}),
    # T2: the temporal lever on the DVS-specific net
    ("DV", "dvs", "full",   "dvsnet",     24, 0, {}),
    # C: same two on DVS
    ("C",  "dvs", "full",   "tiny",       16, 0, {}),
    ("C",  "dvs", "full",   "smallsew",   16, 0, {}),
    # C EXTREME: smallest model on smallest input — the efficiency headline
    ("C",  "sl",  "crop64", "tiny",       16, 0, dict(keep_frac=0.7)),
    ("C",  "dvs", "crop64", "tiny",       16, 0, dict(keep_frac=0.7)),
    # T2: confirm + push
    ("DV", "dvs", "full",   "dvsnet",     24, 1, {}),
    ("DV", "dvs", "full",   "dvsnet",     16, 1, {}),
    # T1: crop64 on DVS + a third SL seed
    ("CR", "dvs", "crop64", "dvsnet",     16, 0, dict(keep_frac=0.7)),
    ("CR", "sl",  "crop64", "ghostsew12", 16, 2, dict(keep_frac=0.7)),
    # C: seeds for error bars on the small models
    ("C",  "sl",  "full",   "tiny",       16, 1, {}),
    ("C",  "dvs", "full",   "tiny",       16, 1, {}),
    ("C",  "sl",  "full",   "smallsew",   16, 1, {}),
    ("C",  "dvs", "full",   "smallsew",   16, 1, {}),
    # T1: last curve point; T2: longest run
    ("KF", "sl",  "hand",   "ghostsew12", 16, 0, dict(keep_frac=0.85)),
    ("DV", "dvs", "full",   "dvsnet",     24, 0, dict(epochs=150)),
]

def run(code, ds, mode, net, T, seed, extra):
    kf = extra.get("keep_frac", 0.5)
    ep = extra.get("epochs", 80)
    tag = f"{code}_{ds}_{mode}_{net}_T{T}_seed{seed}"
    if mode in ("hand", "crop64"):
        tag += f"_keep{kf}"
    if ep != 80:
        tag += f"_e{ep}"
    tag += "_aug"
    if os.path.exists(os.path.join(RESULTS, f"{tag}.json")):
        print(f"SKIP {tag}", flush=True); return
    print(f"\n{'='*70}\nRUN  {tag}\n{'='*70}", flush=True)
    t0 = time.time()
    try:
        a = train_and_eval(code=code, ds=ds, mode=mode, net=net, T=T, seed=seed,
                           epochs=ep, **BASE,
                           **{k: v for k, v in extra.items() if k != "epochs"})
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
                      d.get("keep_frac", 0.5)), []).append((d["test_acc"], d["params"]))
    print(f"\n{'ds':5s}{'arm':8s}{'net':13s}{'T':4s}{'kf':5s}{'n':3s}{'mean':>8s}"
          f"{'std':>7s}{'params':>10s}", flush=True)
    for k in sorted(g):
        a = sorted(x[0] for x in g[k]); pr = g[k][0][1]
        sd = statistics.stdev(a) if len(a) > 1 else 0.0
        print(f"{k[0]:5s}{k[1]:8s}{k[2]:13s}{str(k[3]):4s}{str(k[4]):5s}{len(a):<3}"
              f"{100*statistics.mean(a):>7.1f}%{100*sd:>7.1f}{pr:>10,}", flush=True)
    for ds in ("sl", "dvs"):
        full = [statistics.mean([x[0] for x in v]) for k, v in g.items()
                if k[0] == ds and k[1] == "full"]
        if not full:
            continue
        bf = max(full)
        for arm in ("hand", "crop64"):
            sub = [statistics.mean([x[0] for x in v]) for k, v in g.items()
                   if k[0] == ds and k[1] == arm]
            if sub:
                print(f"{ds.upper()}: best full {100*bf:.1f}% vs best {arm} "
                      f"{100*max(sub):.1f}%  -> GAP {100*(bf-max(sub)):.1f}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    sys.stdout = _Tee(LOG)
    if a.summary:
        summary()
    else:
        for q in QUEUE: run(*q)
        print("\nV6 COMPLETE", flush=True); summary()
