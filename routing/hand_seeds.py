"""Final hand-arm seeds — completes the A/B comparison on ghostsew12.

Full arms are settled: SL 90.2 +- 0.8 (n=4), DVS 91.9 +- 0.2 (n=3).
Hand arms had only 2 seeds each and both first seeds were high draws
(SL 77.2 then 72.5; DVS 90.5 then 86.0), so the gaps are not yet reliable.
"""
import os, sys, time, json, glob, statistics, traceback
import dataset as D
from mark_model import train_and_eval

RESULTS = os.path.join(D.DATA, "results")
LOG = os.path.join(RESULTS, "hand_seeds.log")
COMMON = dict(T=16, patience=20, lr=5e-4, epochs=80, batch=8)
AUG = dict(augment=True, mix_p=0.3, smooth=0.1)

class _Tee:
    def __init__(s, p):
        s.f = open(p, "a", buffering=1, encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x)
    def flush(s): s.o.flush(); s.f.flush()

QUEUE = [("sl", 2), ("dvs", 2), ("sl", 3), ("dvs", 3)]

def run(ds, seed):
    tag = f"SEW_{ds}_hand_ghostsew12_T16_seed{seed}_keep0.5_aug"
    if os.path.exists(os.path.join(RESULTS, f"{tag}.json")):
        print(f"SKIP {tag}", flush=True); return
    print(f"\n{'='*70}\nRUN  {tag}\n{'='*70}", flush=True)
    t0 = time.time()
    try:
        a = train_and_eval(code="SEW", ds=ds, mode="hand", net="ghostsew12",
                           seed=seed, **AUG, **COMMON)
        print(f"DONE {tag}: test {a:.3f} ({(time.time()-t0)/60:.0f} min)", flush=True)
    except Exception:
        print(f"FAILED {tag}\n{traceback.format_exc()}", flush=True)

def summary():
    g = {}
    for p in glob.glob(os.path.join(RESULTS, "SEW_*ghostsew12*.json")):
        d = json.load(open(p))
        g.setdefault((d["dataset"], d["mode"]), []).append(d["test_acc"])
    print(f"\n{'ds':5s}{'arm':6s}{'n':3s}{'mean':>8s}{'std':>7s}   runs", flush=True)
    for k in sorted(g):
        a = sorted(g[k]); sd = statistics.stdev(a) if len(a) > 1 else 0.0
        print(f"{k[0]:5s}{k[1]:6s}{len(a):<3}{100*statistics.mean(a):>7.1f}%{100*sd:>7.1f}"
              f"   {[round(100*x,1) for x in a]}", flush=True)
    for ds in ("sl", "dvs"):
        f, h = g.get((ds, "full")), g.get((ds, "hand"))
        if f and h and len(f) > 1 and len(h) > 1:
            mf, mh = statistics.mean(f), statistics.mean(h)
            se = (statistics.stdev(f)**2/len(f) + statistics.stdev(h)**2/len(h))**0.5
            t = (mf-mh)/se if se else 999
            print(f"\n{ds.upper()} A/B: full {100*mf:.1f}% vs hand {100*mh:.1f}% | "
                  f"gap {100*(mf-mh):.1f} | t={t:.2f} -> "
                  + ("SIGNIFICANT" if abs(t) > 2.4 else "not significant"), flush=True)

if __name__ == "__main__":
    sys.stdout = _Tee(LOG)
    if "--summary" in sys.argv:
        summary()
    else:
        for ds, s in QUEUE: run(ds, s)
        print("\nHAND SEEDS COMPLETE", flush=True); summary()
