"""
routing/explore_v7.py — two new architectures, each with its own control.

DVS: MotionSEW
  DVSGesture's classes ARE directions (right/left wave, cw/ccw arm). T=24 bought
  nothing over T=16, so the temporal information is present but not being read as
  DIRECTION. DDEStem computes it explicitly: x[t] * roll(x[t-1], d) is large where
  content moved in direction d. Four correlations, ZERO parameters.
  MotionGate then scales channels by frame motion energy (2*C params) — unlike the
  temporal attention that lost 11.7 points by reweighting whole timesteps.

  ABLATION (this is the point of the three runs):
    dvsnet        495,059  no DDE, no gate   <- baseline, 92.2 +- 0.8
    motionsew_ng  495,923  DDE only          <- isolates DDE
    motionsew     496,155  DDE + gate        <- isolates the gate
  Parameter counts are within 0.2%, so any difference is the MECHANISM, not size.

C: RecurrentSEW
  Depth is our strongest lever; parameter cuts cost ~7 points. Resolve both by
  SHARING weights across depth — one block applied k times gives k-times-deeper
  computation at 1x parameter cost. ghostsew18 needed 339k for 18 blocks;
  recsew6 gets 18 block applications for 64k.

    python explore_v7.py --summary
"""
import argparse, glob, json, os, statistics, sys, time, traceback
import dataset as D
from mark_model import train_and_eval

RESULTS = os.path.join(D.DATA, "results")
LOG = os.path.join(RESULTS, "explore_v7.log")
BASE = dict(T=16, patience=20, lr=5e-4, epochs=80, batch=8, augment=True,
            mix_p=0.3, smooth=0.1)

class _Tee:
    def __init__(s, p):
        s.f = open(p, "a", buffering=1, encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x)
    def flush(s): s.o.flush(); s.f.flush()

QUEUE = [
    # --- DVS ablation: DDE, then the gate, against the 92.2% baseline ---
    ("V7", "dvs", "full", "motionsew",    0),
    ("V7", "dvs", "full", "motionsew_ng", 0),
    # --- C: weight-shared depth, both datasets ---
    ("V7", "dvs", "full", "recsew6",      0),
    ("V7", "sl",  "full", "recsew6",      0),
    ("V7", "dvs", "full", "recsew_tiny",  0),
    ("V7", "sl",  "full", "recsew_tiny",  0),
    # --- seeds on whatever is worth confirming ---
    ("V7", "dvs", "full", "motionsew",    1),
    ("V7", "dvs", "full", "recsew6",      1),
    ("V7", "sl",  "full", "recsew6",      1),
    ("V7", "dvs", "full", "motionsew12",  0),
    ("V7", "dvs", "full", "recsew4",      0),
    ("V7", "dvs", "full", "motionsew_ng", 1),
    ("V7", "sl",  "full", "recsew_tiny",  1),
    ("V7", "dvs", "full", "recsew_tiny",  1),
]

def run(code, ds, mode, net, seed):
    tag = f"{code}_{ds}_{mode}_{net}_T16_seed{seed}_aug"
    if os.path.exists(os.path.join(RESULTS, f"{tag}.json")):
        print(f"SKIP {tag}", flush=True); return
    print(f"\n{'='*70}\nRUN  {tag}\n{'='*70}", flush=True)
    t0 = time.time()
    try:
        a = train_and_eval(code=code, ds=ds, mode=mode, net=net, seed=seed, **BASE)
        print(f"DONE {tag}: test {a:.3f} ({(time.time()-t0)/60:.0f} min)", flush=True)
    except Exception:
        print(f"FAILED {tag}\n{traceback.format_exc()}", flush=True)

def summary():
    g = {}
    for p in glob.glob(os.path.join(RESULTS, "*.json")):
        if os.path.basename(p).startswith(("ensemble", "run_all", "explore", "hand_seeds")):
            continue
        d = json.load(open(p))
        if d["mode"] != "full" or d.get("T") != 16:
            continue
        g.setdefault((d["dataset"], d["net"]), []).append((d["test_acc"], d["params"]))
    for ds in ("dvs", "sl"):
        sub = {k: v for k, v in g.items() if k[0] == ds}
        if not sub: continue
        print(f"\n=== {ds.upper()} full frame, T=16 ===", flush=True)
        print(f"{'net':16s}{'n':3s}{'mean':>8s}{'std':>7s}{'best':>7s}{'params':>10s}", flush=True)
        for k in sorted(sub, key=lambda k: -statistics.mean([x[0] for x in sub[k]])):
            a = [x[0] for x in sub[k]]
            sd = statistics.stdev(a) if len(a) > 1 else 0.0
            print(f"{k[1]:16s}{len(a):<3}{100*statistics.mean(a):>7.1f}%{100*sd:>7.1f}"
                  f"{100*max(a):>7.1f}{sub[k][0][1]:>10,}", flush=True)
    # the ablation, spelled out
    print("\n=== DVS ABLATION (params within 0.2%, so differences are mechanism) ===",
          flush=True)
    for net, what in (("dvsnet", "baseline (no DDE, no gate)"),
                      ("motionsew_ng", "+ DDE motion channels"),
                      ("motionsew", "+ DDE + motion gate")):
        v = g.get(("dvs", net))
        if v:
            a = [x[0] for x in v]
            sd = statistics.stdev(a) if len(a) > 1 else 0.0
            print(f"  {net:14s} {100*statistics.mean(a):5.1f}% +- {100*sd:4.1f}  n={len(a)}"
                  f"   {what}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    sys.stdout = _Tee(LOG)
    if a.summary:
        summary()
    else:
        for q in QUEUE: run(*q)
        print("\nV7 COMPLETE", flush=True); summary()
