import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK = "#1F2328"; MUT = "#5A6470"; EDGE = "#C6CDD4"; FILL = "#F4F6F8"
ACC_FILL = "#E7F0F9"; ACC_EDGE = "#2F6FA8"

fig_w, fig_h = 6.9, 2.78
fig = plt.figure(figsize=(fig_w, fig_h), dpi=320)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis("off")

labels = [("1  Ingest", "weekly\nfiles"), ("2  Align", "store x SKU\nx week"),
          ("3  Forecast", "units +\nrange"), ("4  Score", "same tests\nevery time"),
          ("5  Rank", "one reason\nper line"), ("6  Emit", "Excel +\ndetail view"),
          ("7  Feedback", "Adopt / Adjust\n/ Ignore")]
n = len(labels); gap = 0.085; left = 0.10
bw = (fig_w - 2 * left - gap * (n - 1)) / n
by, bh = 1.02, 0.74
centers = []
for i, (t, sub) in enumerate(labels):
    x = left + i * (bw + gap); centers.append(x + bw / 2)
    accent = (i == 4)
    ax.add_patch(FancyBboxPatch((x, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.055",
                                linewidth=1.0, edgecolor=ACC_EDGE if accent else EDGE,
                                facecolor=ACC_FILL if accent else FILL, zorder=2))
    ax.text(x + bw / 2, by + bh - 0.20, t, ha="center", va="center", fontsize=7.2,
            color=INK, fontweight="bold" if accent else "normal", zorder=3)
    ax.text(x + bw / 2, by + 0.215, sub, ha="center", va="center", fontsize=5.6,
            color=MUT, zorder=3, linespacing=1.4)

for i in range(n - 1):
    ax.annotate("", xy=(centers[i + 1] - bw / 2 - 0.010, by + bh / 2),
                xytext=(centers[i] + bw / 2 + 0.010, by + bh / 2),
                arrowprops=dict(arrowstyle="-|>", color=MUT, lw=0.9, shrinkA=0,
                                shrinkB=0, mutation_scale=8), zorder=1)

yl = 0.80
ax.plot([centers[6], centers[6]], [by, yl], color=ACC_EDGE, lw=0.9, zorder=1)
ax.plot([centers[6], centers[4]], [yl, yl], color=ACC_EDGE, lw=0.9, zorder=1)
ax.annotate("", xy=(centers[4], by - 0.008), xytext=(centers[4], yl),
            arrowprops=dict(arrowstyle="-|>", color=ACC_EDGE, lw=0.9, shrinkA=0,
                            shrinkB=0, mutation_scale=8), zorder=1)
ax.text((centers[4] + centers[6]) / 2, yl - 0.14,
        "next cycle: thresholds recalibrated from what planners did",
        ha="center", va="center", fontsize=5.7, color=ACC_EDGE, style="italic")


def band(x0, x1, y, txt):
    ax.plot([x0, x1], [y, y], color="#DFE4E9", lw=2.4, solid_capstyle="butt", zorder=1)
    ax.text((x0 + x1) / 2, y + 0.075, txt, ha="center", va="bottom", fontsize=6.1, color=MUT)


band(left, centers[3] + bw / 2, 1.93, "Tuesday night, unattended")
band(centers[4] - bw / 2, left + n * (bw + gap) - gap, 1.93, "Wednesday morning, with the planner")

# storage and the surface a planner opens: the two elements Section 2.1 asks the diagram for
sy, sh = 0.18, 0.34
ax.add_patch(FancyBboxPatch((left, sy), fig_w - 2 * left, sh,
                            boxstyle="round,pad=0,rounding_size=0.045",
                            linewidth=0.8, edgecolor=EDGE, facecolor="#FBFCFD",
                            linestyle=(0, (3, 2)), zorder=2))
ax.text(left + 0.10, sy + sh / 2,
        "stored between cycles:  panel  \u00b7  forecasts  \u00b7  feedback log",
        ha="left", va="center", fontsize=5.8, color=MUT, zorder=3)
ax.text(fig_w - left - 0.10, sy + sh / 2,
        "surface a planner opens:  CSV in Excel",
        ha="right", va="center", fontsize=5.8, color=MUT, zorder=3)

fig.savefig("fig1_architecture.png", dpi=320, facecolor="white")
print("ok")
