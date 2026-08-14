"""
routing/run_all.py
------------------
Runs the whole improvement pipeline unattended and writes a summary table.

  Stage 1  augmentation test        does aug + T=16 + new recipe fix the overfit?
  Stage 2  architecture zoo         which model wins (SL, full frame)
  Stage 3  winner on all four arms  A1, B1, A2, B2 final numbers
  Stage 4  ensemble                 max accuracy from the saved checkpoints

Safe to stop and restart: a run whose results JSON already exists is SKIPPED,
and a crash in one run does not stop the rest.

    python run_all.py                # everything
    python run_all.py --stage 1      # one stage
    python run_all.py --summary      # just print the table of what exists
"""

import argparse
import glob
import json
import os
import sys
import time
import traceback

import dataset as D
from mark_model import train_and_eval

RESULTS = os.path.join(D.DATA, "results")
LOG = os.path.join(D.DATA, "results", "run_all.log")

T = 16
EPOCHS = 40
AUG = dict(augment=True, mix_p=0.3, smooth=0.1)


class _Tee:
    """Mirror everything printed (including per-epoch training lines) into the log
    file, so progress is watchable in VS Code even when the run is detached."""

    def __init__(self, path):
        self.f = open(path, "a", buffering=1, encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()


def log(msg):
    print(msg, flush=True)      # Tee sends it to the log file too


def tag_for(code, ds, mode, net, keep_frac=0.5, seed=0, aug=True):
    t = f"{code}_{ds}_{mode}_{net}_T{T}_seed{seed}"
    if mode in ("hand", "crop64"):
        t += f"_keep{keep_frac}"
    if aug:
        t += "_aug"
    return t


def run(code, ds, mode, net, **kw):
    """One experiment, skipped if already done, never fatal."""
    tag = tag_for(code, ds, mode, net, kw.get("keep_frac", 0.5), kw.get("seed", 0))
    path = os.path.join(RESULTS, f"{tag}.json")
    if os.path.exists(path):
        acc = json.load(open(path))["test_acc"]
        log(f"SKIP {tag} (done, test {acc:.3f})")
        return acc
    log(f"\n{'='*70}\nRUN  {tag}\n{'='*70}")
    t0 = time.time()
    try:
        acc = train_and_eval(code=code, ds=ds, mode=mode, net=net,
                             T=T, epochs=EPOCHS, **AUG, **kw)
        log(f"DONE {tag}: test {acc:.3f}  ({(time.time()-t0)/60:.0f} min)")
        return acc
    except Exception:
        log(f"FAILED {tag}\n{traceback.format_exc()}")
        return None


# ----------------------------------------------------------------------
def stage1():
    """Does augmentation + T=16 + the new recipe fix the overfitting?"""
    log("\n########## STAGE 1 — augmentation test ##########")
    run("A1", "sl", "full", "standard")
    run("B1", "sl", "hand", "standard")


ZOO = ["base", "plif", "attn", "deep", "ghost", "wide", "combo"]


def stage2():
    """Which architecture wins, on SL full frame."""
    log("\n########## STAGE 2 — architecture zoo ##########")
    for net in ZOO:
        run("Z", "sl", "full", net)


def best_zoo_net():
    best, acc = None, -1
    for net in ZOO:
        p = os.path.join(RESULTS, f"{tag_for('Z','sl','full',net)}.json")
        if os.path.exists(p):
            a = json.load(open(p))["test_acc"]
            if a > acc:
                best, acc = net, a
    return best, acc


def stage3():
    """Apply the zoo winner to all four arms."""
    log("\n########## STAGE 3 — winner on all arms ##########")
    net, acc = best_zoo_net()
    if net is None:
        log("no zoo results yet — run stage 2 first"); return
    log(f"zoo winner: {net} (SL full test {acc:.3f})")
    for code, ds, mode in (("A1", "sl", "full"), ("B1", "sl", "hand"),
                           ("A2", "dvs", "full"), ("B2", "dvs", "hand")):
        run(code, ds, mode, net)


def stage4():
    """Ensemble the saved checkpoints per arm."""
    log("\n########## STAGE 4 — ensemble ##########")
    import subprocess
    import sys
    for ds, mode in (("sl", "full"), ("sl", "hand")):
        log(f"\n--- ensemble {ds} {mode} ---")
        r = subprocess.run([sys.executable, "ensemble.py", "--ds", ds,
                            "--mode", mode, "--T", str(T)],
                           capture_output=True, text=True)
        log(r.stdout or r.stderr)


# ----------------------------------------------------------------------
def summary():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        if os.path.basename(p).startswith(("ensemble", "run_all")):
            continue
        d = json.load(open(p))
        rows.append((d.get("code", "?"), d["dataset"], d["mode"], d["net"],
                     d.get("T"), d.get("augment", False), d["params"],
                     d["best_val_acc"], d["test_acc"], d.get("overfit_gap")))
    rows.sort(key=lambda r: (r[1], r[2], -r[8]))
    log(f"\n{'code':5s}{'ds':5s}{'mode':7s}{'net':10s}{'T':4s}{'aug':5s}"
        f"{'params':>9s}{'val':>7s}{'test':>7s}{'gap':>7s}")
    log("-" * 72)
    for c, ds, m, n, t, a, p, v, te, g in rows:
        log(f"{c:5s}{ds:5s}{m:7s}{n:10s}{str(t):4s}{'Y' if a else 'N':5s}"
            f"{p:>9,}{v:>7.3f}{te:>7.3f}{(f'{g:.3f}' if g is not None else '-'):>7s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0, help="1-4, or 0 for all")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    sys.stdout = _Tee(LOG)          # everything from here on lands in the log file
    if a.summary:
        summary()
    else:
        t0 = time.time()
        stages = [stage1, stage2, stage3, stage4]
        for i, fn in enumerate(stages, 1):
            if a.stage in (0, i):
                fn()
        log(f"\nTOTAL {(time.time()-t0)/60:.0f} min")
        summary()
