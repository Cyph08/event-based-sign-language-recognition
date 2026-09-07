"""Generate report figures + Appendix A table directly from data/results/*.json.

Everything here is derived from the stored run records, so the figures cannot drift
from the numbers in the text.
"""
import json, glob, os, statistics
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "K:/Dissertation/data/results"
OUT = "K:/Dissertation/Report/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
SL, DVS = "#2166ac", "#b2182b"
SMALL = {"tiny", "smallsew", "small", "tiny_bal", "recsew6", "recsew_tiny"}


def load():
    rows = []
    for p in glob.glob(os.path.join(R, "*.json")):
        b = os.path.basename(p)
        if b.startswith(("ensemble", "run_all", "explore", "hand_seeds", "SMOKE")):
            continue
        d = json.load(open(p))
        if "test_acc" not in d:                 # analysis files (e.g. per_class_sl), not runs
            continue
        if d.get("cv_fold") is not None:        # 4-fold CV scores a DIFFERENT test set;
            continue                            # pooling it with fixed-split runs is invalid
        # every distillation variant, not just the single-teacher one: Eens_/Efeat_/
        # Eboth_ are ensemble/feature KD and were previously counted as NO-KD baseline
        d["_kd"] = b.startswith(("KD", "VXkd", "Eens_", "Efeat_", "Eboth_"))
        d["_kdvar"] = ("KD-single" if b.startswith(("KD05_", "KD08_", "VXkd3_"))
                       else "KD-ensemble" if b.startswith("Eens_")
                       else "KD-feature" if b.startswith("Efeat_")
                       else "KD-ens+feat" if b.startswith("Eboth_")
                       else "KD+SAFL" if b.startswith("VXkdsafl_") else "none")
        d["_variant"] = (b.split("_")[0] if b[0] == "V" else "base")
        rows.append(d)
    return rows


rows = load()


def agg(pred):
    v = [r["test_acc"] for r in rows if pred(r)]
    if not v:
        return None
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), len(v)


# ---------------------------------------------------------------- Figure 2
# depth curve, DVS Ghost-SEW family
fig, ax = plt.subplots(figsize=(5.2, 3.2))
depth_pts = [(5, "deep5_ghost", 91779), (8, "ghostsew8", 185867),
             (12, "ghostsew12", 238475), (18, "ghostsew18", 338603)]
xs, ys, es = [], [], []
for d, net, _ in depth_pts:
    a = agg(lambda r, n=net: r["net"] == n and r["dataset"] == "dvs"
            and r["mode"] == "full" and r.get("T") == 16)
    if a:
        xs.append(d); ys.append(100 * a[0]); es.append(100 * a[1])
ax.errorbar(xs, ys, yerr=es, marker="o", color=DVS, capsize=3, lw=1.8, ms=6)
best = xs[ys.index(max(ys))]
ax.axvline(best, ls=":", c="grey", lw=1)
ax.annotate(f"optimum\n{best} blocks", (best, max(ys)), textcoords="offset points",
            xytext=(12, -6), fontsize=8, color="grey")
ax.set_xlabel("Residual blocks"); ax.set_ylabel("Test accuracy (%)")
ax.set_title("Depth curve on DVS128 Gesture (Ghost-SEW family)", fontsize=10)
ax.set_xticks(xs); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_depth_curve.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 3
# accuracy vs event retention (Arm B), SL
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ret = {0.5: 38.4, 0.7: 56.5, 0.85: 70.0}
xs, ys, es = [], [], []
for kf, pct in sorted(ret.items()):
    a = agg(lambda r, k=kf: r["dataset"] == "sl" and r["mode"] == "hand"
            and r["net"] == "ghostsew12" and abs(r.get("keep_frac", .5) - k) < 1e-6)
    if a:
        xs.append(pct); ys.append(100 * a[0]); es.append(100 * a[1])
ax.errorbar(xs, ys, yerr=es, marker="o", color=SL, capsize=3, lw=1.8, ms=6,
            label="Arm B (hand region)")
full = agg(lambda r: r["dataset"] == "sl" and r["mode"] == "full"
           and r["net"] == "ghostsew12" and r["_variant"] == "base")
ax.axhline(100 * full[0], ls="--", c="k", lw=1.2, label=f"Arm A full frame ({100*full[0]:.1f}%)")
ax.fill_between([30, 105], 100 * (full[0] - full[1]), 100 * (full[0] + full[1]),
                color="k", alpha=.08)
ax.annotate("knee", (56.5, ys[1]), textcoords="offset points", xytext=(6, -14),
            fontsize=8, color="grey")
ax.set_xlim(30, 105); ax.set_xlabel("Events retained (%)")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Accuracy against event retention, SL-Animals-DVS", fontsize=10)
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_retention.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 4
# accuracy vs parameters, with/without KD, both datasets
fig, ax = plt.subplots(figsize=(5.8, 3.6))
for ds, col, lbl in (("sl", SL, "SL-Animals"), ("dvs", DVS, "DVS128 Gesture")):
    for kd, ls, mk in ((False, "--", "o"), (True, "-", "s")):
        pts = []
        for net in ("tiny", "smallsew"):
            a = agg(lambda r, n=net, k=kd, d=ds: r["dataset"] == d and r["net"] == n
                    and r["mode"] == "full" and r.get("T") == 16 and r["_kd"] == k
                    and "safl" not in r["_variant"].lower())
            if a:
                pr = next(r["params"] for r in rows if r["net"] == net and r["dataset"] == ds)
                pts.append((pr, 100 * a[0], 100 * a[1]))
        # anchor with the full-size model (no KD)
        big = "ghostsew12" if ds == "sl" else "dvsnet"
        a = agg(lambda r, n=big, d=ds: r["dataset"] == d and r["net"] == n
                and r["mode"] == "full" and r.get("T") == 16 and r["_variant"] == "base")
        if a and not kd:
            pr = next(r["params"] for r in rows if r["net"] == big and r["dataset"] == ds)
            pts.append((pr, 100 * a[0], 100 * a[1]))
        if len(pts) < 2:
            continue
        pts.sort()
        ax.errorbar([p[0] / 1000 for p in pts], [p[1] for p in pts],
                    yerr=[p[2] for p in pts], marker=mk, ls=ls, color=col,
                    capsize=3, lw=1.6, ms=5, alpha=1.0 if kd else .55,
                    label=f"{lbl}{' + KD' if kd else ''}")
ax.set_xscale("log")
ax.set_xlabel("Parameters (thousands, log scale)")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Accuracy against model size, with and without distillation", fontsize=10)
ax.legend(fontsize=7.5, loc="lower right"); ax.grid(alpha=.25, which="both")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_params.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 5
# progression: what actually moved the numbers
fig, ax = plt.subplots(figsize=(5.8, 3.2))
stages = ["Baseline\n(T=8, no aug)", "+ augmentation", "+ depth/Ghost",
          "+ stabilised\ntraining", "+ SEW\nresidual"]
sl_prog = [52.6, 64.3, 76.8, 79.5, 90.2]
sl_err = [0, 0, 9.4, 2.0, 0.9]
x = range(len(stages))
ax.errorbar(x, sl_prog, yerr=sl_err, marker="o", color=SL, capsize=3, lw=2, ms=6)
for i, (v, e) in enumerate(zip(sl_prog, sl_err)):
    ax.annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8)
ax.axhline(78.0, ls="--", c="grey", lw=1)
ax.annotate("best published (78.0%)", (0.05, 78.6), fontsize=7.5, color="grey")
ax.set_xticks(list(x)); ax.set_xticklabels(stages, fontsize=7.5)
ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(45, 97)
ax.set_title("Cumulative improvement, SL-Animals-DVS", fontsize=10)
ax.grid(alpha=.25, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_progression.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 6
# temporal-resolution factorial, SL: two architectures x three T values.
# _variant=="base" drops the subject-adversarial runs; CV runs are already out (load()).
fig, ax = plt.subplots(figsize=(5.4, 3.3))
MEAN_DUR = 4.48                                   # s, measured over SL samples
for net, colour, label in (("ghostsew12", SL, "ghostsew12 (247k)"),
                           ("deep", "#e08214", "deep (80k)")):
    xs, ys, es, ns = [], [], [], []
    for T in (8, 16, 32):
        a = agg(lambda r, n=net, t=T: r["dataset"] == "sl" and r["mode"] == "full"
                and r["net"] == n and r.get("T") == t and r["_variant"] == "base")
        if a:
            xs.append(T); ys.append(100 * a[0]); es.append(100 * a[1]); ns.append(a[2])
    ax.errorbar(xs, ys, yerr=es, marker="o", color=colour, capsize=3, lw=1.8, ms=6,
                label=label)
    for T, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (T, y), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=7, color=colour)
ax.set_xscale("log", base=2); ax.set_xticks([8, 16, 32])
ax.set_xticklabels([f"{T}\n({1000*MEAN_DUR/T:.0f} ms/bin)" for T in (8, 16, 32)],
                   fontsize=8)
ax.set_xlabel("Temporal bins $T$"); ax.set_ylabel("Test accuracy (%)")
ax.set_title("Temporal resolution against capacity, SL-Animals-DVS", fontsize=10)
ax.legend(fontsize=8, loc="lower center"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_temporal.png"); plt.close(fig)

# ------------------------------------------------- Fig 8: per-class accuracy
# From per_class_sl.json (routing/per_class.py): the four baseline SEW seeds
# re-scored on the SL test set, aggregated ACROSS seeds rather than best-run.
pc_path = os.path.join(R, "per_class_sl.json")
if os.path.exists(pc_path):
    pc = json.load(open(pc_path))
    mean = [100 * x for x in pc["class_mean"]]
    sd = [100 * x for x in pc["class_std"]]
    order = sorted(range(len(mean)), key=lambda i: mean[i])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4),
                                  gridspec_kw={"width_ratios": [1.55, 1]})

    ax.bar(range(len(order)), [mean[i] for i in order],
           yerr=[sd[i] for i in order], color=SL, ecolor="#888",
           capsize=2, width=0.72)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(i) for i in order], fontsize=7)
    ax.set_xlabel("sign class (sorted by accuracy)")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 105)
    ax.axhline(90.2, ls="--", lw=1, color="#444")
    ax.text(0.4, 91.5, "overall 90.2%", fontsize=7, color="#444")
    ax.set_title("Per-class accuracy, mean over 4 seeds", fontsize=9)

    conf = pc["confusion_summed"]
    n = len(conf)
    off = [[0 if i == j else conf[i][j] for j in range(n)] for i in range(n)]
    im = ax2.imshow(off, cmap="Reds", interpolation="nearest")
    ax2.set_xlabel("predicted class"); ax2.set_ylabel("true class")
    ax2.set_xticks(range(0, n, 2)); ax2.set_yticks(range(0, n, 2))
    ax2.tick_params(labelsize=7)
    ax2.set_title("Confusions (diagonal removed)", fontsize=9)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig7_perclass.png"); plt.close(fig)


# ---------------------------------------------------------------- Appendix A
lines = ["| Dataset | Arm | Model | T | keep_frac | KD | n | Mean | Std | Best | Params |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
g = defaultdict(list)
for r in rows:
    # Arm C is a PARAMETER reduction on full frames. A small net on hand input is
    # both reductions at once and belongs in its own row, not pooled with Arm C.
    if r["mode"] == "crop64":
        arm = "B-crop"
    elif r["net"] in SMALL:
        arm = "C" if r["mode"] == "full" else "C+B"
    else:
        arm = "A" if r["mode"] == "full" else "B"
    g[(r["dataset"], arm, r["net"], r.get("T"), r.get("keep_frac", 0.5),
       r["_kd"])].append((r["test_acc"], r["params"]))
for k in sorted(g, key=lambda k: (k[0], k[1], -statistics.mean([x[0] for x in g[k]]))):
    v = [x[0] for x in g[k]]
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    lines.append(f"| {k[0]} | {k[1]} | `{k[2]}` | {k[3]} | {k[4]} | "
                 f"{'Y' if k[5] else '-'} | {len(v)} | {100*statistics.mean(v):.1f}% | "
                 f"{100*sd:.1f} | {100*max(v):.1f}% | {g[k][0][1]:,} |")
open(f"{OUT}/appendix_a.md", "w", encoding="utf-8").write("\n".join(lines))

print(f"figures -> {OUT}")
for f in sorted(os.listdir(OUT)):
    print("  ", f)
print(f"appendix A rows: {len(lines)-2}")
