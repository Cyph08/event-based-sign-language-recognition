"""Figure 1 for the Introduction: what the data is, and what the three arms do."""
import sys, json, os
sys.path.insert(0, "K:/Dissertation/routing")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dataset as D
from mark_model import RecognitionDataset

plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "font.family": "DejaVu Sans"})
sp = json.load(open(os.path.join(D.DATA, "splits", "sl_animals.json")))
full = RecognitionDataset("sl", sp["test"], "full", 16, 0.5, "train")
hand = RecognitionDataset("sl", sp["test"], "hand", 16, 0.7, "train")

idx = 3
xf, _ = full[idx]; xh, _ = hand[idx]
xf, xh = np.asarray(xf), np.asarray(xh)

fig = plt.figure(figsize=(6.6, 3.5))
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1], hspace=.32, wspace=.12)

bins = [2, 6, 9, 12]
for j, b in enumerate(bins):
    ax = fig.add_subplot(gs[0, j])
    im = xf[b].sum(0)
    ax.imshow(im, cmap="inferno", vmax=max(im.max(), 1e-6))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t={b}", fontsize=7, pad=2)
    if j == 0:
        ax.set_ylabel("Arm A\nfull frame", fontsize=7.5)

for j, b in enumerate(bins):
    ax = fig.add_subplot(gs[1, j])
    im = xh[b].sum(0)
    ax.imshow(im, cmap="inferno", vmax=max(im.max(), 1e-6))
    ax.set_xticks([]); ax.set_yticks([])
    if j == 0:
        ax.set_ylabel("Arm B\nhand region", fontsize=7.5)

# right-hand panel: the reduction summary
ax = fig.add_subplot(gs[:, 4]); ax.axis("off")
ret = 100 * float(xh.sum()) / max(float(xf.sum()), 1e-9)
ax.text(0, .93, "The question", fontsize=8.5, weight="bold", transform=ax.transAxes)
ax.text(0, .60,
        "An event camera has\nalready discarded\neverything static.\n\n"
        "Is the remaining\nstream itself\nredundant?\n\n"
        f"Arm B keeps\n{ret:.0f}% of events\nfrom the same sign.",
        fontsize=7.4, va="top", transform=ax.transAxes, linespacing=1.5)
ax.add_patch(plt.Rectangle((-.05, -.02), 1.12, 1.02, fill=False, ec="#999",
                           lw=.8, transform=ax.transAxes, clip_on=False))

fig.suptitle("Event data for one sign, and the two input conditions compared",
             fontsize=9.5, y=.99)
fig.savefig("figures/fig1_intro.png", bbox_inches="tight")
print(f"saved figures/fig1_intro.png  (retention {ret:.1f}%)")
