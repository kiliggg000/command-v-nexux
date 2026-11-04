"""Two figures for the shortlisting deck, drawn from the same artefacts the
technical document reports: forecast_weekly.parquet and the holdout reports."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator

INK   = "#0B1F3A"
TEAL  = "#00A6A6"
MUTED = "#586980"
GREY  = "#9FB2C4"
RED   = "#B4483C"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GREY,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# ---------------------------------------------------------------- figure 1
f = pd.read_parquet("forecast_weekly.parquet")
f = f[f.week_index <= 51]                      # 12 scored weeks, week-52 stub out
bias = (f.groupby("store_id")
          .apply(lambda d: (d.y.sum() - d.mu.sum()) / d.y.sum(), include_groups=False) * 100)

edges = [-np.inf, 0, 2, 4, 6, 8, 10, 12, np.inf]
labels = ["under 0", "0–2", "2–4", "4–6", "6–8", "8–10", "10–12", "12+"]
counts = pd.cut(bias, bins=edges, labels=labels, right=False).value_counts().reindex(labels)

fig, ax = plt.subplots(figsize=(6.0, 3.35), dpi=240)
colors = [GREY] + [TEAL] * 7
bars = ax.bar(range(len(labels)), counts.values, color=colors, width=0.72, zorder=3)

for i, (c, b) in enumerate(zip(counts.values, bars)):
    ax.text(b.get_x() + b.get_width() / 2, c + 0.7, str(int(c)),
            ha="center", va="bottom", fontsize=10.5, color=INK,
            fontweight="bold" if i else "normal")

ax.axhline(0, color=GREY, lw=0.8, zorder=2)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=10)
ax.set_xlabel("forecast bias per store, % of actual sales", fontsize=10.5, labelpad=6)
ax.set_ylabel("stores", fontsize=10.5, labelpad=6)
ax.set_ylim(0, max(counts.values) * 1.30)
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.grid(axis="y", color="#E3EBF2", lw=0.8, zorder=0)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)

ax.text(-0.42, max(counts.values) * 1.20, "130 of 140 stores under-forecast",
        ha="left", va="center", fontsize=11.5, color=TEAL, fontweight="bold")
fig.tight_layout(pad=0.4)
fig.savefig("fig_store_bias.png", dpi=240)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
ks, ys = [0, 4, 8], []
for name in ("holdout_report.json", "holdout_report_k4.json", "holdout_report_k8.json"):
    ys.append(json.load(open(name))["held"]["wape"])
seen = json.load(open("holdout_report.json"))["seen"]["wape"]
bar = round(seen * 1.05, 4)

fig, ax = plt.subplots(figsize=(6.0, 3.35), dpi=240)
ax.axhspan(0, bar, color="#EAF6F6", zorder=0)
ax.plot(ks, ys, color=TEAL, lw=2.4, marker="o", ms=9, mfc="white", mew=2.4,
        zorder=4, clip_on=False)
for x, y in zip(ks, ys):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(-2 if x == 8 else 0, 13),
                ha="center", fontsize=10.5, color=INK, fontweight="bold")

ax.axhline(bar, color=RED, lw=1.3, ls=(0, (5, 3)), zorder=2)
ax.text(-0.35, bar - 0.006, f"bar {bar:.3f}: within 5% of a store we can see",
        ha="left", va="top", fontsize=9.5, color=RED)
ax.axhline(1.0, color=GREY, lw=1.0, ls=(0, (2, 2)), zorder=2)
ax.text(8.6, 1.003, "predicting zero", ha="right", va="bottom", fontsize=9.5, color=MUTED)

ax.set_xlim(-0.55, 8.75)
ax.set_ylim(0.80, 1.055)
ax.set_xticks(ks)
ax.set_xticklabels(["none", "4 weeks", "8 weeks"], fontsize=10.5)
ax.set_xlabel("weeks of that store's own history", fontsize=10.5, labelpad=6)
ax.set_ylabel("WAPE on the held-out stores", fontsize=10.5, labelpad=6)
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.grid(axis="y", color="#E3EBF2", lw=0.8, zorder=1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig("fig_history_weeks.png", dpi=240)
plt.close(fig)
print("bias mean %.2f  counts %s" % (bias.mean(), list(counts.values)))
print("k", ks, "wape", ys, "bar", bar)
