"""
routing/explore_v11.py — overnight: extend the ONE technique that worked.

CONTEXT. Seven interventions have been tested under matched conditions:
  forward-pass changes  PLIF -12.3 | attn -11.7 | crop64 -16.0 | recsew -6.7
                        motionsew -2.8 | capacity realloc -1.3        0/6
  training-signal       SAFL 0.0 (n=3, looked like +1.3 at n=2)       0/1
                        knowledge distillation  +2.5 / +3.2           1/1
So the only productive direction is the training signal, and distillation is the
only member of that class that has held up. Two extensions were planned earlier
and never run; this queue runs them.

  ENSEMBLE TEACHER  average the distributions of the top-3 checkpoints selected by
                    VALIDATION (never test). Any single teacher has idiosyncratic
                    confusions from its own initialisation; averaging cancels those
                    and keeps what the teachers agree on. Distillation's largest
                    measured effect here was VARIANCE reduction (tiny: +-1.9 ->
                    +-0.3), so a lower-variance target is the natural next step.

  FEATURE KD        logit KD transfers WHAT the teacher decides; an MSE term on
                    pre-head features also transfers HOW it represents the input.
                    A 1x1 projection bridges the width mismatch and is DISCARDED
                    after training, so reported parameter counts are unchanged.

EXCLUDED BY INSTRUCTION: the large SL model (ghostsew12 on SL-Animals) is settled
at 90.2% +- 0.9 (n=4) and is not touched. Everything else is in scope.

    python explore_v11.py --summary
"""
import argparse, glob, json, os, statistics, sys, time, traceback
import dataset as D
from mark_model import train_and_eval

RESULTS = os.path.join(D.DATA, "results")
CKPT = os.path.join(D.DATA, "ckpt")
LOG = os.path.join(RESULTS, "explore_v11.log")
BASE = dict(T=16, patience=20, lr=5e-4, epochs=80, batch=8, augment=True,
            mix_p=0.3, smooth=0.1)

# top-3 by VALIDATION accuracy, T=16 only (must match the student's input)
ENS = {
    "sl": [os.path.join(CKPT, f) for f in (
        "SEW_sl_full_ghostsew12_T16_seed3_aug.pt",     # val .947
        "SEW_sl_full_ghostsew12_T16_seed1_aug.pt",     # val .930
        "SEW_sl_full_ghostsew12_T16_seed2_aug.pt")],   # val .930
    "dvs": [os.path.join(CKPT, f) for f in (
        "V9subj_dvs_full_dvsnet_T16_seed0_aug.pt",     # val .975
        "SEW_dvs_full_ghostsew12_T16_seed0_aug.pt",    # val .969
        "DV_dvs_full_dvsnet_T16_seed1_aug.pt")],       # val .969
}
SINGLE = {"sl": [ENS["sl"][0]], "dvs": [ENS["dvs"][1]]}


class _Tee:
    def __init__(s, p):
        s.f = open(p, "a", buffering=1, encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x)
    def flush(s): s.o.flush(); s.f.flush()


# (tag, ds, net, seed, kwargs)
QUEUE = [
    # --- ensemble teacher vs the single-teacher control we already have ---
    ("ens",  "dvs", "tiny",       0, dict(teacher_ckpt=ENS["dvs"], kd_w=0.5)),
    ("ens",  "sl",  "tiny",       0, dict(teacher_ckpt=ENS["sl"],  kd_w=0.5)),
    ("ens",  "dvs", "smallsew",   0, dict(teacher_ckpt=ENS["dvs"], kd_w=0.5)),
    ("ens",  "sl",  "smallsew",   0, dict(teacher_ckpt=ENS["sl"],  kd_w=0.5)),
    # --- feature KD on top of the single teacher (isolates the feature term) ---
    ("feat", "dvs", "tiny",       0, dict(teacher_ckpt=SINGLE["dvs"], kd_w=0.5, feat_w=0.1)),
    ("feat", "sl",  "tiny",       0, dict(teacher_ckpt=SINGLE["sl"],  kd_w=0.5, feat_w=0.1)),
    # --- both together, on the smallest students where KD pays most ---
    ("both", "dvs", "tiny",       0, dict(teacher_ckpt=ENS["dvs"], kd_w=0.5, feat_w=0.1)),
    ("both", "sl",  "tiny",       0, dict(teacher_ckpt=ENS["sl"],  kd_w=0.5, feat_w=0.1)),
    # --- seeds on whichever variants are worth confirming ---
    ("ens",  "dvs", "tiny",       1, dict(teacher_ckpt=ENS["dvs"], kd_w=0.5)),
    ("ens",  "sl",  "tiny",       1, dict(teacher_ckpt=ENS["sl"],  kd_w=0.5)),
    ("ens",  "dvs", "smallsew",   1, dict(teacher_ckpt=ENS["dvs"], kd_w=0.5)),
    ("ens",  "sl",  "smallsew",   1, dict(teacher_ckpt=ENS["sl"],  kd_w=0.5)),
    ("both", "dvs", "tiny",       1, dict(teacher_ckpt=ENS["dvs"], kd_w=0.5, feat_w=0.1)),
    ("both", "sl",  "tiny",       1, dict(teacher_ckpt=ENS["sl"],  kd_w=0.5, feat_w=0.1)),
    # --- DVS large model with the ensemble teacher (SL large is excluded) ---
    ("ens",  "dvs", "ghostsew12", 0, dict(teacher_ckpt=ENS["dvs"], kd_w=0.3)),
]


def run(tag, ds, net, seed, kw):
    code = f"E{tag}"
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
    g = {}
    for p in glob.glob(os.path.join(RESULTS, "*.json")):
        b = os.path.basename(p)
        if b.startswith(("ensemble", "run_all", "explore", "hand_seeds", "SMOKE")):
            continue
        d = json.load(open(p))
        if d["mode"] != "full" or d.get("T") != 16 or d["net"] not in (
                "tiny", "smallsew", "ghostsew12", "dvsnet"):
            continue
        if b.startswith("Eens"):    v = "KD-ensemble"
        elif b.startswith("Efeat"): v = "KD-feature"
        elif b.startswith("Eboth"): v = "KD-ens+feat"
        elif b.startswith(("KD", "VXkd")): v = "KD-single"
        elif b.startswith(("V9", "VX")):   v = "other"
        else: v = "no KD"
        if v == "other":
            continue
        g.setdefault((d["dataset"], d["net"], v), []).append(d["test_acc"])
    for ds in ("sl", "dvs"):
        print(f"\n=== {ds.upper()} — distillation variants ===", flush=True)
        print(f"{'model':12s}{'variant':14s}{'n':3s}{'mean':>8s}{'std':>7s}", flush=True)
        for net in ("tiny", "smallsew", "ghostsew12", "dvsnet"):
            for v in ("no KD", "KD-single", "KD-ensemble", "KD-feature", "KD-ens+feat"):
                a = g.get((ds, net, v))
                if not a:
                    continue
                sd = statistics.stdev(a) if len(a) > 1 else 0.0
                print(f"{net:12s}{v:14s}{len(a):<3}{100*statistics.mean(a):>7.1f}%"
                      f"{100*sd:>7.1f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    sys.stdout = _Tee(LOG)
    if a.summary:
        summary()
    else:
        t0 = time.time()
        for q in QUEUE:
            if time.time() - t0 > 12.5 * 3600:
                print("\nTIME BUDGET REACHED", flush=True); break
            run(*q)
        print(f"\nV11 COMPLETE — {(time.time()-t0)/60:.0f} min", flush=True)
        summary()
