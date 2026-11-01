import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#1F2328"; MUT = "#5A6470"; GRID = "#E6E9EC"
NEUTRAL = "#D3D7DB"; BASE = "#2F6FA8"; RED = "#A6402F"

# Values transcribed from the original figure's own printed labels.
# (label, cell WAPE, bias in percent, style)
rows = [
    ("Drop store effect",            1.077, -7.4, "red"),
    ("No store x SKU effect",        1.073, -7.7, "red"),
    ("Drop seasonality",             0.857, -6.9, "n"),
    ("No pooling",                   0.851, -4.8, "n"),
    ("Baseline (current)",           0.851, -4.7, "base"),
    ("Drop log_price",               0.851, -4.7, "n"),
    ("No β shrinkage",            0.851, -4.7, "n"),
    ("Drop log_store_weight",        0.851, -4.7, "n"),
    ("Strong cell shrinkage (λ=2)", 0.851, 2.7, "n"),
    ("Add linear trend",             0.832,  4.9, "red"),
]

names = [r[0] for r in rows]
wape = [r[1] for r in rows]
bias = [r[2] / 100.0 for r in rows]
colors = {"n": NEUTRAL, "base": BASE, "red": RED}
kind = [r[3] for r in rows]
y = np.arange(len(rows))[::-1]

fig = plt.figure(figsize=(6.9, 2.62), dpi=320)
gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.05], wspace=0.07,
                      left=0.255, right=0.975, top=0.965, bottom=0.20)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

ax1.barh(y, wape, height=0.62, color=[colors[k] for k in kind], edgecolor="none", zorder=2)
for yy, v in zip(y, wape):
    ax1.text(v + 0.013, yy, f"{v:.3f}", va="center", ha="left", fontsize=6.0, color=INK, zorder=6)
ax1.set_xlim(0, 1.20)
ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=6.6, color=INK)
ax1.set_xlabel("cell WAPE   (lower is better)", fontsize=6.4, color=MUT, labelpad=4)
ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.tick_params(axis="x", labelsize=5.8, colors=MUT, length=2)

ax2.barh(y, bias, height=0.62, color=[colors[k] for k in kind], edgecolor="none", zorder=2)
ax2.axvline(0, color="#9AA3AC", lw=0.7, zorder=3)
for yy, v in zip(y, bias):
    pos = v >= 0
    ax2.text(v + (0.004 if pos else -0.004), yy, f"{v*100:+.1f}%", va="center",
             ha="left" if pos else "right", fontsize=5.9, color=INK, zorder=6)
ax2.set_xlim(-0.142, 0.122)
ax2.set_yticks(y); ax2.set_yticklabels([])
ax2.set_xlabel("bias, weeks 40-43", fontsize=6.4, color=MUT, labelpad=4)
ax2.set_xticks([-0.10, -0.05, 0, 0.05, 0.10])
ax2.tick_params(axis="x", labelsize=5.8, colors=MUT, length=2)

for ax in (ax1, ax2):
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#C6CDD4")
    ax.tick_params(axis="y", length=0, pad=4)

fig.savefig("fig3_ablation.png", dpi=320, facecolor="white")
print("ok")
