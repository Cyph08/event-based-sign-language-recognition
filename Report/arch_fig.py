import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

def box(x, y, w, h, txt, fc, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec="#444", lw=.9))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, lw=1, color="#444"))

# main pipeline
box(0.15, 4.6, 1.7, .8, "Events\n[T=16, 2, 128, 128]", "#dbe9f6")
arrow(1.85, 5.0, 2.35, 5.0)
box(2.35, 4.6, 2.2, .8, "STEM\nConv→BN→LIF→2×MaxPool\n128→32", "#fde8d8", 7)
arrow(4.55, 5.0, 5.05, 5.0)
box(5.05, 4.6, 2.3, .8, "12 × Ghost-SEW block\nchannels ×2 at 4, 8\n32→16→8", "#e8f4e0", 7)
arrow(7.35, 5.0, 7.85, 5.0)
box(7.85, 4.6, 2.0, .8, "AvgPool(4)→Flatten\nLinear→classes\nmean over T", "#dbe9f6", 7)

# exploded block
ax.text(5.0, 3.85, "Ghost-SEW residual block (detail)", fontsize=8.5, ha="center")
y = 2.3
box(0.9, y, 1.25, .7, "GhostConv", "#e8f4e0")
arrow(2.15, y+.35, 2.5, y+.35)
box(2.5, y, .8, .7, "BN", "#f2f2f2")
arrow(3.3, y+.35, 3.65, y+.35)
box(3.65, y, .8, .7, "LIF", "#ffe9ec")
arrow(4.45, y+.35, 4.8, y+.35)
box(4.8, y, 1.25, .7, "GhostConv", "#e8f4e0")
arrow(6.05, y+.35, 6.4, y+.35)
box(6.4, y, .8, .7, "BN", "#f2f2f2")
arrow(7.2, y+.35, 7.55, y+.35)
box(7.55, y, .8, .7, "LIF", "#ffe9ec")
arrow(8.35, y+.35, 8.75, y+.35)
ax.add_patch(plt.Circle((8.95, y+.35), .19, fc="w", ec="#444", lw=1))
ax.text(8.95, y+.35, "+", ha="center", va="center", fontsize=11)

# shortcut path
ax.plot([0.55, 0.55, 8.95, 8.95], [y+.35, y+1.25, y+1.25, y+.62],
        color="#b2182b", lw=1.2, ls="--")
ax.text(4.7, y+1.35, "shortcut added to OUTPUT SPIKES (SEW)",
        ha="center", fontsize=7.5, color="#b2182b")
arrow(0.15, y+.35, 0.85, y+.35)
ax.text(9.4, y+.35, "out", fontsize=7.5, va="center")

ax.text(0.15, 1.55, "GhostConv: half the maps from a real conv, half from a cheap depthwise conv (~2× fewer parameters)",
        fontsize=7, color="#333")
ax.text(0.15, 1.2, "LIF: leaky integrate-and-fire neuron, arctangent surrogate gradient",
        fontsize=7, color="#333")
ax.text(0.15, 0.85, "Total: 247k parameters (SL-Animals, 19 classes)", fontsize=7, color="#333")

fig.tight_layout()
fig.savefig("figures/fig1_architecture.png", bbox_inches="tight")
print("saved figures/fig1_architecture.png")
