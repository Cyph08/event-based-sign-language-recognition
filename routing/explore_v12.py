"""
routing/explore_v12.py — temporal-resolution factorial.

WHY. Draft 1 claimed "temporal resolution substitutes for architectural capacity but does
not add to it." A supervisor review found the evidence insufficient, correctly. It was:

    SL   `deep`   T16 69.0% -> T32 74.3%      n=1 EACH
    DVS  `dvsnet` T16 -> T24 (+0.3)           DIFFERENT dataset, DIFFERENT T, n=2

Architecture, dataset and T value are all confounded and one arm is a single seed. The
alternative reading — that the effect belongs to the capacity and temporal-integration
properties of the particular final network rather than to temporal redundancy in general —
cannot be distinguished from ours with that data.

DESIGN. One dataset (SL-Animals, genuinely subject-independent splits), two architectures
of very different capacity, three T values. Everything else held fixed.

                        T=8      T=16          T=32
    deep        (80k)   run x2   have 1, +1    have 1, +1
    ghostsew12 (247k)   run x2   have n=4      run x2

WHAT IT DISTINGUISHES.
  weak net gains with T, strong net does not  -> capacity-substitution supported
  both gain                                    -> T carries information both can use;
                                                  Draft 1 claim withdrawn
  neither gains                                -> the earlier +5.3 was seed noise

CONTEXT FOR INTERPRETATION. Measured sample durations: SL-Animals mean 4.48 s
(range 1.24-9.30). So T=8/16/32 correspond to roughly 560/280/140 ms per bin. DVSGesture
averages 6.65 s, i.e. 415 ms per bin at T=16 — the same T is NOT the same temporal
resolution across datasets, which the report now states explicitly.

NOTE. The sl_full_T32 tensor cache was deleted during a clean-up, so the first T=32 run
rebuilds it (adds roughly 30-45 min once).

    python explore_v12.py --summary
"""
import argparse, glob, json, os, statistics, sys, time, traceback
import dataset as D
from mark_model import train_and_eval

RESULTS = os.path.join(D.DATA, "results")
LOG = os.path.join(RESULTS, "explore_v12.log")
BASE = dict(patience=20, lr=5e-4, epochs=80, batch=8, augment=True,
            mix_p=0.3, smooth=0.1)
MEAN_DUR = {"sl": 4.48, "dvs": 6.65}          # seconds, measured over 400 samples


class _Tee:
    def __init__(s, p):
        s.f = open(p, "a", buffering=1, encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x)
    def flush(s): s.o.flush(); s.f.flush()


# (net, T, seed) — cheap runs first so the factorial fills in even if the night is cut short
QUEUE = [
    ("ghostsew12", 8,  0), ("ghostsew12", 8,  1),   # strong net, coarse time
    ("deep",       8,  0), ("deep",       8,  1),   # weak net, coarse time
    ("deep",      16,  1),                          # second seed at the reference point
    ("deep",      32,  1),                          # second seed, weak net fine time
    ("ghostsew12", 32, 0), ("ghostsew12", 32, 1),   # strong net, fine time (expensive)
    # round 2 (2 Sep): both nets peaked at T16 in round 1, but T8/T32 rest on n=2 each.
    # Two more seeds per edge point firms up the U-shape before it goes in the report.
    ("deep",       8,  2), ("deep",       8,  3),
    ("ghostsew12", 8,  2), ("ghostsew12", 8,  3),
    ("deep",      32,  2), ("deep",      32,  3),
    ("ghostsew12", 32, 2), ("ghostsew12", 32, 3),
]


def run(net, T, seed):
    tag = f"TF_sl_full_{net}_T{T}_seed{seed}_aug"
    if os.path.exists(os.path.join(RESULTS, f"{tag}.json")):
        print(f"SKIP {tag}", flush=True); return
    print(f"\n{'='*70}\nRUN  {tag}   [{time.strftime('%H:%M')}]   "
          f"bin width ~{1000*MEAN_DUR['sl']/T:.0f} ms\n{'='*70}", flush=True)
    t0 = time.time()
    try:
        a = train_and_eval(code="TF", ds="sl", mode="full", net=net, T=T, seed=seed, **BASE)
        print(f"DONE {tag}: test {a:.3f} ({(time.time()-t0)/60:.0f} min)", flush=True)
    except Exception:
        print(f"FAILED {tag}\n{traceback.format_exc()}", flush=True)


def summary():
    """The factorial, plus every prior T result so the comparison is complete."""
    g = {}
    for p in glob.glob(os.path.join(RESULTS, "*.json")):
        b = os.path.basename(p)
        if b.startswith(("ensemble", "run_all", "explore", "hand_seeds", "SMOKE")):
            continue
        d = json.load(open(p))
        if "test_acc" not in d:            # analysis files, not run records
            continue
        if d.get("mode") != "full" or d.get("net") not in ("deep", "ghostsew12"):
            continue
        if d.get("dataset") != "sl":
            continue
        # A T cell must pool runs that differ ONLY in T. Two things break that:
        # cv_fold runs score a different test set (4-fold CV), and safl_lam runs
        # carry an extra adversarial loss. Both land on ghostsew12/T16 and would
        # silently corrupt the reference cell this whole factorial is measured against.
        if d.get("cv_fold") is not None or d.get("safl_lam"):
            continue
        g.setdefault((d["net"], d.get("T")), []).append(d["test_acc"])

    print("\n=== SL-ANIMALS temporal factorial (bin width from mean duration 4.48 s) ===",
          flush=True)
    print(f"{'net':13s}{'params':>9s}{'T':>4s}{'bin ms':>8s}{'n':>4s}{'mean':>9s}{'std':>7s}",
          flush=True)
    par = {"deep": 80195, "ghostsew12": 246675}
    for net in ("deep", "ghostsew12"):
        for T in (8, 16, 32):
            a = g.get((net, T))
            if not a:
                print(f"{net:13s}{par[net]:>9,}{T:>4}{1000*MEAN_DUR['sl']/T:>8.0f}"
                      f"{'-':>4}{'not run':>9s}", flush=True)
                continue
            sd = statistics.stdev(a) if len(a) > 1 else 0.0
            print(f"{net:13s}{par[net]:>9,}{T:>4}{1000*MEAN_DUR['sl']/T:>8.0f}"
                  f"{len(a):>4}{100*statistics.mean(a):>8.1f}%{100*sd:>7.1f}", flush=True)

    print("\n=== effect of T within each architecture ===", flush=True)
    for net in ("deep", "ghostsew12"):
        pts = {T: g.get((net, T)) for T in (8, 16, 32)}
        base = pts.get(16)
        if not base:
            continue
        b = statistics.mean(base)
        for T in (8, 32):
            a = pts.get(T)
            if a:
                print(f"  {net:12s} T{T:<3} vs T16: {100*(statistics.mean(a)-b):+5.1f} pts "
                      f"(n={len(a)} vs n={len(base)})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    sys.stdout = _Tee(LOG)
    if a.summary:
        summary()
    else:
        t0 = time.time()
        for q in QUEUE:
            if time.time() - t0 > 48.0 * 3600:
                print("\nTIME BUDGET REACHED", flush=True); break
            run(*q)
        print(f"\nV12 COMPLETE — {(time.time()-t0)/60:.0f} min", flush=True)
        summary()
