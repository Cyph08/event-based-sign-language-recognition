"""
routing/explore_v8.py — Section C via knowledge distillation.

WHY THIS, AFTER TWO ARCHITECTURE FAILURES:
  motionsew (explicit motion encoding)  -2.8 vs baseline
  recsew6   (weight-shared depth)       -6.7 vs a SMALLER plain model
Both tried to make the small model smarter. Neither worked, and the reason is the
same one that killed PLIF (-12.3) and temporal attention (-11.7): adding machinery
to a capacity-starved spiking net costs more than the machinery returns.

So stop changing the student and change the TRAINING SIGNAL instead.

A small net trained on hard labels gets ~1 bit per sample (779 samples on SL). The
teacher's softened distribution says WHICH classes are confusable — "this is a dog,
but it looks 30% like a cat" — which is exactly the structure a low-capacity model
cannot infer alone. The student's architecture is untouched, so the parameter count
in the C table stays honest.

Teachers (best available ghostsew12):
  SL  91.2%  SEW_sl_full_ghostsew12_T16_seed3_aug.pt
  DVS 92.0%  SEW_dvs_full_ghostsew12_T16_seed0_aug.pt

CONTROLS: the same students already exist WITHOUT distillation —
  tiny      SL 83.0 +- 2.5   DVS 84.7 +- 4.0
  smallsew  SL 88.3 +- 1.7   DVS 88.6 +- 2.7
so the KD effect is a direct like-for-like comparison at identical parameter count.

    python explore_v8.py --summary
"""
import argparse, glob, json, os, statistics, sys, time, traceback
import dataset as D
from mark_model import train_and_eval

RESULTS = os.path.join(D.DATA, "results")
CKPT = os.path.join(D.DATA, "ckpt")
LOG = os.path.join(RESULTS, "explore_v8.log")
BASE = dict(T=16, patience=20, lr=5e-4, epochs=80, batch=8, augment=True,
            mix_p=0.3, smooth=0.1)
TEACHER = {
    "sl":  os.path.join(CKPT, "SEW_sl_full_ghostsew12_T16_seed3_aug.pt"),
    "dvs": os.path.join(CKPT, "SEW_dvs_full_ghostsew12_T16_seed0_aug.pt"),
}

class _Tee:
    def __init__(s, p):
        s.f = open(p, "a", buffering=1, encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x)
    def flush(s): s.o.flush(); s.f.flush()

# (ds, net, seed, kd_w)
QUEUE = [
    ("sl",  "tiny",     0, 0.5),
    ("dvs", "tiny",     0, 0.5),
    ("sl",  "smallsew", 0, 0.5),
    ("dvs", "smallsew", 0, 0.5),
    ("sl",  "tiny",     1, 0.5),
    ("dvs", "tiny",     1, 0.5),
    ("sl",  "smallsew", 1, 0.5),
    ("dvs", "smallsew", 1, 0.5),
    # does a heavier teacher weight help a very small student?
    ("sl",  "tiny",     0, 0.8),
    ("dvs", "tiny",     0, 0.8),
    # error bar on the motionsew negative result, so it is reportable
    ("_ng", "motionsew_ng", 1, 0.0),
]

def run(ds, net, seed, kd_w):
    if ds == "_ng":                      # the no-KD control run for the negative result
        tag = f"V7_dvs_full_{net}_T16_seed{seed}_aug"
        if os.path.exists(os.path.join(RESULTS, f"{tag}.json")):
            print(f"SKIP {tag}", flush=True); return
        print(f"\n{'='*70}\nRUN  {tag}\n{'='*70}", flush=True)
        t0 = time.time()
        try:
            a = train_and_eval(code="V7", ds="dvs", mode="full", net=net,
                               seed=seed, **BASE)
            print(f"DONE {tag}: test {a:.3f} ({(time.time()-t0)/60:.0f} min)", flush=True)
        except Exception:
            print(f"FAILED {tag}\n{traceback.format_exc()}", flush=True)
        return

    w = str(kd_w).replace(".", "")
    tag = f"KD{w}_{ds}_full_{net}_T16_seed{seed}_aug"
    if os.path.exists(os.path.join(RESULTS, f"{tag}.json")):
        print(f"SKIP {tag}", flush=True); return
    print(f"\n{'='*70}\nRUN  {tag}   (teacher kd_w={kd_w})\n{'='*70}", flush=True)
    t0 = time.time()
    try:
        a = train_and_eval(code=f"KD{w}", ds=ds, mode="full", net=net, seed=seed,
                           teacher_ckpt=TEACHER[ds], kd_w=kd_w, **BASE)
        print(f"DONE {tag}: test {a:.3f} ({(time.time()-t0)/60:.0f} min)", flush=True)
    except Exception:
        print(f"FAILED {tag}\n{traceback.format_exc()}", flush=True)

def summary():
    g = {}
    for p in glob.glob(os.path.join(RESULTS, "*.json")):
        b = os.path.basename(p)
        if b.startswith(("ensemble", "run_all", "explore", "hand_seeds")): continue
        d = json.load(open(p))
        if d["mode"] != "full" or d.get("T") != 16: continue
        kd = b.startswith("KD")
        g.setdefault((d["dataset"], d["net"], kd), []).append((d["test_acc"], d["params"]))
    for ds in ("sl", "dvs"):
        print(f"\n=== {ds.upper()} — distillation effect ===", flush=True)
        print(f"{'net':12s}{'KD':4s}{'n':3s}{'mean':>8s}{'std':>7s}{'params':>10s}", flush=True)
        for net in ("tiny", "smallsew", "ghostsew12"):
            for kd in (False, True):
                v = g.get((ds, net, kd))
                if not v: continue
                a = [x[0] for x in v]
                sd = statistics.stdev(a) if len(a) > 1 else 0.0
                print(f"{net:12s}{'Y' if kd else 'N':4s}{len(a):<3}"
                      f"{100*statistics.mean(a):>7.1f}%{100*sd:>7.1f}{v[0][1]:>10,}", flush=True)
            base = g.get((ds, net, False)); dist = g.get((ds, net, True))
            if base and dist:
                d0 = statistics.mean([x[0] for x in base])
                d1 = statistics.mean([x[0] for x in dist])
                print(f"  -> {net}: KD effect {100*(d1-d0):+.1f} pts", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    sys.stdout = _Tee(LOG)
    if a.summary:
        summary()
    else:
        for q in QUEUE: run(*q)
        print("\nV8 COMPLETE", flush=True); summary()
