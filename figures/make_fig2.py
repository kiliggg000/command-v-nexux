"""Redraw Figure 3: D2 demand against the promotion calendar it was built from.

The repository figure carries its own title and clips the y-axis label at the top
edge, which duplicates the caption and looks careless in print. This version plots
the same two series with margins that survive A4 at 6.15 inches.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

SRC = Path(__file__).resolve().parent.parent

INK = "#1F2328"; MUT = "#5A6470"; BAR = "#9DB7CF"; LINE = "#C0392B"; DASH = "#8894A0"

p = pd.read_parquet(SRC / "panel_weekly.parquet",
                    columns=["week_index", "qty_d1", "qty_d2", "depth_pct"])
g = p.groupby("week_index").agg(d2=("qty_d2", "mean"),
                                d1=("qty_d1", "mean"),
                                depth=("depth_pct", "mean")).reset_index()
# week 52 is the two-day stub, not a comparable week
g = g[g["week_index"] < 52]

fig, ax = plt.subplots(figsize=(6.15, 2.45), dpi=320)
fig.subplots_adjust(left=0.085, right=0.885, top=0.90, bottom=0.155)

x = g["week_index"].to_numpy()
ax.bar(x, g["d2"], width=0.74, color=BAR, edgecolor="none", zorder=2,
       label="D2, demand with structure")
ax.plot(x, g["d1"], color=DASH, lw=1.2, ls=(0, (2.4, 1.6)), zorder=3,
        label="D1, the same mean with the structure removed")
ax.set_xlabel("week of 2024", fontsize=6.8, color=MUT, labelpad=3)
ax.set_ylabel("units per store and product", fontsize=6.8, color=MUT, labelpad=4)
ax.tick_params(labelsize=6.2, colors=MUT, length=2.5, pad=2)
ax.set_xlim(-1, 52)
ax.set_ylim(0, max(g["d2"]) * 1.34)
for s in ("top",):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom", "right"):
    ax.spines[s].set_color("#D5DBE1")
ax.grid(axis="y", color="#EDF0F3", lw=0.7, zorder=0)
ax.set_axisbelow(True)

ax2 = ax.twinx()
ax2.plot(x, g["depth"] * 100, color=LINE, lw=1.3, zorder=4,
         label="mean promotion depth in the price archive")
ax2.set_ylabel("promotion depth, per cent", fontsize=6.8, color=LINE, labelpad=5)
ax2.tick_params(labelsize=6.2, colors=LINE, length=2.5, pad=2)
ax2.set_ylim(0, 40)
ax2.spines["top"].set_visible(False)
for s in ("left", "bottom", "right"):
    ax2.spines[s].set_color("#D5DBE1")

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 1.115),
          ncol=3, frameon=False, fontsize=5.9, handlelength=1.6,
          columnspacing=1.4, handletextpad=0.5)

r = np.corrcoef(g["d2"], g["depth"])[0, 1]
ax.annotate(f"the two series move together: weekly correlation {r:.2f}",
            xy=(0.985, 0.955), xycoords="axes fraction", ha="right", va="top",
            fontsize=6.0, color=MUT, style="italic")

fig.savefig("fig2_demand.png", dpi=320, facecolor="white")
print("ok", len(g), "weeks")
