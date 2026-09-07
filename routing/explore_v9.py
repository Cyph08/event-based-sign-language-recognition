"""
routing/explore_v9.py — customized dvsnet / tiny, every change traced to a measurement.

DIAGNOSIS DRIVING ALL OF THIS
  Every model UNDERFITS (train acc BELOW val acc) yet loses 2.3-4.9 points from
  val to TEST. Validation shares subjects with training; test does not. So the
  bottleneck is SIGNER-SPECIFIC FEATURES, not capacity and not overfitting.
  => report val->test drop, not just accuracy. That is the number to move.

TRACK RECORD THIS PLAN RESPECTS
  forward-pass machinery : PLIF -12.3 | attn -11.7 | crop64 -16.0
                           recsew -6.7 | motionsew -2.8      0/5 worked
  training-signal changes: distillation +2.1 +2.7 +1.2 +3.0   4/4 worked
  => three of the four changes below are TRAINING-TIME ONLY and vanish at inference.

THE FOUR CHANGES
  SAFL       gradient-reversal subject classifier -> trunk cannot encode WHO is
             signing. Aux head discarded after training; parameter counts unchanged.
  subj-aug   tempo jitter (0.8-1.25x) + body-size jitter (0.85-1.15x), with erase
             reduced 0.25->0.15 so the stack is RETARGETED, not made heavier.
  light-aug  control: is the current stack over-regularising? (erase 0.15 only)
  *_bal      capacity REALLOCATION. dvsnet keeps 68% of parameters in 4 blocks at
             8x8; _bal caps channel growth and pools later, so more blocks run at
             32x32/16x16. Matched budgets: 479,291 vs 495,059 (96.8%),
             37,531 vs 38,587 (97.3%) — the comparison isolates allocation.

    python explore_v9.py --summary
"""
import argparse, glob, json, os, statistics, sys, time, traceback
import dataset as D
from mark_model import train_and_eval

RESULTS = os.path.join(D.DATA, "results")
CKPT = os.path.join(D.DATA, "ckpt")
LOG = os.path.join(RESULTS, "explore_v9.log")
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

# (tag, ds, net, seed, kwargs)
QUEUE = [
    # STAGE 1 — isolate each change on DVS against dvsnet 92.2% +- 0.8
    ("safl",   "dvs", "dvsnet",     0, dict(safl_lam=0.3)),
    ("subj",   "dvs", "dvsnet",     0, dict(aug_variant="subj")),
    ("light",  "dvs", "dvsnet",     0, dict(aug_variant="light")),
    ("bal",    "dvs", "dvsnet_bal", 0, {}),
    ("saflsub", "dvs", "dvsnet",    0, dict(safl_lam=0.3, aug_variant="subj")),

    # STAGE 2 — SL has 41 training subjects, the strongest case for SAFL
    ("safl",   "sl",  "ghostsew12", 0, dict(safl_lam=0.3)),
    ("subj",   "sl",  "ghostsew12", 0, dict(aug_variant="subj")),

    # STAGE 3 — tiny: stack the winner with distillation (proven +2.1/+3.2)
    ("kdsafl", "dvs", "tiny",       0, dict(safl_lam=0.3,
                                            teacher_ckpt=TEACHER["dvs"], kd_w=0.5)),
    ("kdsafl", "sl",  "tiny",       0, dict(safl_lam=0.3,
                                            teacher_ckpt=TEACHER["sl"], kd_w=0.5)),
    ("bal",    "dvs", "tiny_bal",   0, {}),
    ("bal",    "sl",  "tiny_bal",   0, {}),

    # STAGE 4 — second seeds on the DVS changes
    ("safl",   "dvs", "dvsnet",     1, dict(safl_lam=0.3)),
    ("subj",   "dvs", "dvsnet",     1, dict(aug_variant="subj")),
    ("bal",    "dvs", "dvsnet_bal", 1, {}),
    ("safl",   "sl",  "ghostsew12", 1, dict(safl_lam=0.3)),
]

def run(tag, ds, net, seed, kw):
    code = f"V9{tag}"
    name = f"{code}_{ds}_full_{net}_T16_seed{seed}_aug"
    if os.path.exists(os.path.join(RESULTS, f"{name}.json")):
        print(f"SKIP {name}", flush=True); return
    print(f"\n{'='*70}\nRUN  {name}\n{'='*70}", flush=True)
    t0 = time.time()
    try:
        a = train_and_eval(code=code, ds=ds, mode="full", net=net, seed=seed,
                           **BASE, **kw)
        print(f"DONE {name}: test {a:.3f} ({(time.time()-t0)/60:.0f} min)", flush=True)
    except Exception:
        print(f"FAILED {name}\n{traceback.format_exc()}", flush=True)

def summary():
    rows = []
    for p in glob.glob(os.path.join(RESULTS, "*.json")):
        b = os.path.basename(p)
        if b.startswith(("ensemble", "run_all", "explore", "hand_seeds", "SMOKE")):
            continue
        d = json.load(open(p))
        if d["mode"] != "full" or d.get("T") != 16:
            continue
        variant = (b.split("_")[0][2:] if b.startswith("V9") else
                   ("KD" if b.startswith("KD") else "base"))
        rows.append((d["dataset"], d["net"], variant, d["test_acc"],
                     d.get("val_test_drop"), d["params"]))
    for ds in ("dvs", "sl"):
        sub = [r for r in rows if r[0] == ds]
        if not sub: continue
        g = {}
        for r in sub:
            g.setdefault((r[1], r[2]), []).append((r[3], r[4], r[5]))
        print(f"\n=== {ds.upper()} — test accuracy AND val->test drop ===", flush=True)
        print(f"{'net':13s}{'variant':10s}{'n':3s}{'test':>8s}{'std':>6s}"
              f"{'v->t drop':>11s}{'params':>10s}", flush=True)
        for k in sorted(g, key=lambda k: -statistics.mean([x[0] for x in g[k]])):
            v = g[k]; a = [x[0] for x in v]
            drops = [x[1] for x in v if x[1] is not None]
            sd = statistics.stdev(a) if len(a) > 1 else 0.0
            dr = f"{100*statistics.mean(drops):+.1f}" if drops else "  -  "
            print(f"{k[0]:13s}{k[1]:10s}{len(a):<3}{100*statistics.mean(a):>7.1f}%"
                  f"{100*sd:>6.1f}{dr:>11s}{v[0][2]:>10,}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    sys.stdout = _Tee(LOG)
    if a.summary:
        summary()
    else:
        for q in QUEUE: run(*q)
        print("\nV9 COMPLETE", flush=True); summary()
