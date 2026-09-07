"""Re-derive every headline number in Final.md from data/results/*.json and
compare against what the report prints. Run before every submission build."""
import json, glob, os, re, statistics as st
from collections import defaultdict

R = "K:/Dissertation/data/results"
SKIP = ("ensemble", "run_all", "explore", "hand_seeds", "SMOKE", "per_class")

def variant(b):
    if b.startswith("Eens_"):  return "KD"
    if b.startswith("Efeat_"): return "KD"
    if b.startswith("Eboth_"): return "KD"
    if b.startswith(("KD05_", "KD08_", "VXkd3_")): return "KD"
    if b.startswith("VXkdsafl_"): return "KD"
    return "noKD"

runs = []
for p in glob.glob(os.path.join(R, "*.json")):
    b = os.path.basename(p)[:-5]
    if b.startswith(SKIP): continue
    d = json.load(open(p))
    if "test_acc" not in d: continue
    d["_b"] = b; d["_kd"] = variant(b) == "KD"
    d["_safl"] = bool(d.get("safl_lam") or 0)
    d["_augvar"] = d.get("aug_variant")
    runs.append(d)

def sel(**kw):
    out = []
    for d in runs:
        if d.get("cv_fold") is not None: continue
        ok = all(d.get(k) == v for k, v in kw.items() if not k.startswith("_"))
        ok = ok and all(d.get(k) == v for k, v in kw.items() if k.startswith("_"))
        if ok: out.append(d["test_acc"])
    return sorted(out)

def ms(v):
    return (100*st.mean(v), 100*st.stdev(v) if len(v) > 1 else 0.0, len(v))

CHECKS = []
def chk(label, v, exp_m, exp_s, exp_n):
    if not v:
        CHECKS.append((label, "NO DATA", "")); return
    m, s, n = ms(v)
    got = f"{m:.1f} ± {s:.1f} (n={n})"
    want = f"{exp_m:.1f} ± {exp_s:.1f} (n={exp_n})"
    ok = abs(m-exp_m) < 0.06 and abs(s-exp_s) < 0.06 and n == exp_n
    CHECKS.append((label, "OK" if ok else "MISMATCH", f"report {want} | data {got}"))

base = dict(mode="full", T=16, keep_frac=0.5, _kd=False, _safl=False, _augvar=None)
chk("T5  SL ghostsew12", sel(dataset="sl", net="ghostsew12", **base), 90.2, 0.9, 4)
chk("T5  DVS dvsnet",    sel(dataset="dvs", net="dvsnet", **base), 92.2, 0.8, 2)
chk("T5  DVS ghostsew12",sel(dataset="dvs", net="ghostsew12", **base), 91.9, 0.2, 3)

hb = dict(mode="hand", net="ghostsew12", T=16, _kd=False, _safl=False, _augvar=None)
chk("T11 SL hand 0.5",  sel(dataset="sl", keep_frac=0.5, **hb), 77.8, 4.1, 4)
chk("T11 SL hand 0.7",  sel(dataset="sl", keep_frac=0.7, **hb), 82.7, 2.9, 2)
chk("T11 SL hand 0.85", sel(dataset="sl", keep_frac=0.85, **hb), 81.3, 2.5, 2)
chk("T11 DVS hand 0.5", sel(dataset="dvs", keep_frac=0.5, **hb), 87.1, 2.6, 4)

for ds, net, m, s, n in (("sl","smallsew",89.5,2.3,3), ("sl","tiny",82.7,1.9,3),
                         ("dvs","smallsew",88.6,2.7,2), ("dvs","tiny",84.7,4.0,2)):
    chk(f"T13 {ds} {net} noKD", sel(dataset=ds, net=net, mode="full", T=16,
        keep_frac=0.5, _kd=False, _safl=False, _augvar=None), m, s, n)
for ds, net, m, s, n in (("sl","smallsew",89.2,1.8,6), ("sl","tiny",84.0,1.1,9),
                         ("dvs","smallsew",91.1,1.1,5), ("dvs","tiny",86.8,2.9,9)):
    chk(f"T13 {ds} {net} KD pooled", [d["test_acc"] for d in runs
        if d.get("cv_fold") is None and d["dataset"]==ds and d["net"]==net
        and d["mode"]=="full" and d.get("T")==16 and d["_kd"]], m, s, n)

tf = dict(dataset="sl", mode="full", keep_frac=0.5, _kd=False, _safl=False, _augvar=None)
for net, T, m, s, n in (("deep",8,82.0,3.4,4), ("deep",16,74.3,7.4,2), ("deep",32,77.0,3.5,4),
                        ("ghostsew12",8,85.7,2.6,4), ("ghostsew12",16,90.2,0.9,4),
                        ("ghostsew12",32,88.6,3.8,4)):
    chk(f"T16 {net} T={T}", sel(net=net, T=T, **tf), m, s, n)

folds = defaultdict(list)
for d in runs:
    if d.get("cv_fold") is not None and d["net"] == "ghostsew12":
        folds[d["cv_fold"]].append(d["test_acc"])
if folds:
    fm = [st.mean(folds[f]) for f in sorted(folds)]
    CHECKS.append(("T7  CV cross-validated",
        "OK" if abs(100*st.mean(fm)-87.6) < 0.06 and abs(100*st.stdev(fm)-1.5) < 0.06 else "MISMATCH",
        f"report 87.6 ± 1.5 (4 folds) | data {100*st.mean(fm):.1f} ± {100*st.stdev(fm):.1f} ({len(fm)} folds)"))

w = max(len(c[0]) for c in CHECKS)
bad = 0
for label, status, detail in CHECKS:
    if status != "OK": bad += 1
    print(f"[{status:8s}] {label:{w}s}  {detail}")
print(f"\n{len(CHECKS)-bad}/{len(CHECKS)} checks pass")
