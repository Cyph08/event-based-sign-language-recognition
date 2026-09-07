"""
routing/explore_v10.py — overnight to 10:00, ordered for a thesis draft.

Deadline-driven. ~12 h available, ~70 min/run, so ~10 runs. Ordered so that if it
is cut short, the most citable results are already in.

PRIORITIES
  1. Confirm SAFL — the only new thing that gained accuracy (DVS 93.6 vs 92.2+-0.8,
     SL 91.2 vs 90.2+-0.9). Both are single-seed; six single-seed results have
     reversed in this project, so seeds decide whether it goes in the draft at all.
  2. Finish SL SECTION C — author flagged it as missing. Results exist but most
     are n=2; add seeds and the best-technique combination.
  3. Stack the winners on the small models (KD + SAFL).

DROPPED (measured failures, not worth the GPU hours before a deadline)
  dvsnet_bal 90.9% (vs 92.2 baseline, and the WORST val->test drop at 6.0)
  -> so tiny_bal and the bal seed are cut too; capacity reallocation is the sixth
     forward-pass change to fail (PLIF, attn, crop64, recsew, motionsew, bal).
  subject-aug seeds: +1.0 alone but WIDENED the val->test gap; lower value than SAFL.

    python explore_v10.py --summary     # full thesis table, both datasets
"""
import argparse, glob, json, os, statistics, sys, time, traceback
import dataset as D
from mark_model import train_and_eval

RESULTS = os.path.join(D.DATA, "results")
CKPT = os.path.join(D.DATA, "ckpt")
LOG = os.path.join(RESULTS, "explore_v10.log")
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

# (tag, ds, net, seed, kwargs) — strict priority order
QUEUE = [
    # 1. does SAFL survive a second seed? (both datasets, the new headline)
    ("safl",    "sl",  "ghostsew12", 1, dict(safl_lam=0.3)),
    ("safl",    "dvs", "dvsnet",     1, dict(safl_lam=0.3)),
    # 2. SL section C — best small model available: KD + SAFL stacked
    ("kdsafl",  "sl",  "smallsew",   0, dict(safl_lam=0.3,
                                             teacher_ckpt=TEACHER["sl"], kd_w=0.5)),
    ("kdsafl",  "sl",  "tiny",       0, dict(safl_lam=0.3,
                                             teacher_ckpt=TEACHER["sl"], kd_w=0.5)),
    # 3. SL section C error bars — third seeds on the existing best
    ("kd3",     "sl",  "smallsew",   2, dict(teacher_ckpt=TEACHER["sl"], kd_w=0.5)),
    ("c3",      "sl",  "smallsew",   2, {}),
    ("c3",      "sl",  "tiny",       2, {}),
    # 4. same stack on DVS section C
    ("kdsafl",  "dvs", "smallsew",   0, dict(safl_lam=0.3,
                                             teacher_ckpt=TEACHER["dvs"], kd_w=0.5)),
    ("kdsafl",  "dvs", "tiny",       0, dict(safl_lam=0.3,
                                             teacher_ckpt=TEACHER["dvs"], kd_w=0.5)),
    # 5. third SAFL seeds if time allows
    ("safl",    "sl",  "ghostsew12", 2, dict(safl_lam=0.3)),
    ("safl",    "dvs", "dvsnet",     2, dict(safl_lam=0.3)),
]

def run(tag, ds, net, seed, kw):
    code = f"VX{tag}"
    name = f"{code}_{ds}_full_{net}_T16_seed{seed}_aug"
    if os.path.exists(os.path.join(RESULTS, f"{name}.json")):
        print(f"SKIP {name}", flush=True); return
    print(f"\n{'='*70}\nRUN  {name}   [{time.strftime('%H:%M')}]\n{'='*70}", flush=True)
    t0 = time.time()
    try:
        a = train_and_eval(code=code, ds=ds, mode="full", net=net, seed=seed,
                           **BASE, **kw)
        print(f"DONE {name}: test {a:.3f} ({(time.time()-t0)/60:.0f} min)", flush=True)
    except Exception:
        print(f"FAILED {name}\n{traceback.format_exc()}", flush=True)

def summary():
    """Full thesis table: every full-frame T=16 model, both datasets."""
    rows = []
    for p in glob.glob(os.path.join(RESULTS, "*.json")):
        b = os.path.basename(p)
        if b.startswith(("ensemble", "run_all", "explore", "hand_seeds", "SMOKE")):
            continue
        d = json.load(open(p))
        if d["mode"] != "full" or d.get("T") != 16:
            continue
        if b.startswith("VX") or b.startswith("V9"):
            variant = b.split("_")[0][2:]
        elif b.startswith("KD"):
            variant = "KD"
        else:
            variant = "base"
        rows.append((d["dataset"], d["net"], variant, d["test_acc"],
                     d.get("val_test_drop"), d["params"]))
    for ds, name in (("sl", "SL-ANIMALS (19 cls, chance 5.3%)"),
                     ("dvs", "DVSGESTURE (11 cls, chance 9.1%)")):
        g = {}
        for r in [x for x in rows if x[0] == ds]:
            g.setdefault((r[1], r[2]), []).append((r[3], r[4], r[5]))
        if not g:
            continue
        print(f"\n{'='*72}\n{name}\n{'='*72}", flush=True)
        print(f"{'model':13s}{'variant':9s}{'n':3s}{'mean':>8s}{'std':>6s}"
              f"{'best':>7s}{'v->t':>7s}{'params':>10s}", flush=True)
        for k in sorted(g, key=lambda k: -statistics.mean([x[0] for x in g[k]])):
            v = g[k]; a = sorted(x[0] for x in v)
            dr = [x[1] for x in v if x[1] is not None]
            sd = statistics.stdev(a) if len(a) > 1 else 0.0
            drs = f"{100*statistics.mean(dr):+.1f}" if dr else "   -"
            print(f"{k[0]:13s}{k[1]:9s}{len(a):<3}{100*statistics.mean(a):>7.1f}%"
                  f"{100*sd:>6.1f}{100*max(a):>7.1f}{drs:>7s}{v[0][2]:>10,}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    sys.stdout = _Tee(LOG)
    if a.summary:
        summary()
    else:
        t0 = time.time()
        for q in QUEUE:
            if time.time() - t0 > 11.2 * 3600:      # stop in time for 10:00
                print("\nTIME BUDGET REACHED — stopping before the deadline", flush=True)
                break
            run(*q)
        print(f"\nV10 COMPLETE — {(time.time()-t0)/60:.0f} min", flush=True)
        summary()
