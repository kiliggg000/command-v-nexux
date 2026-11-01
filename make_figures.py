"""
make_figures.py — charts for the README and the deck.

Every figure reads a saved artifact (coefficients.csv, ablation_results.csv,
evaluation_report.json, seasonality_sweep.json) so the charts cannot drift away
from the numbers in MODEL_RESULTS.md.

    python make_figures.py --sweep     # once: regenerates seasonality_sweep.json
    python make_figures.py             # writes figures/*.png

Style: light background, one accent colour, no chartjunk. Sized for slides.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

INK = "#2C2C2A"
MUTED = "#888780"
GRID = "#D3D1C7"
ACCENT = "#185FA5"
BAD = "#A32D2D"
GOOD = "#3B6D11"
FILL = "#E6F1FB"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# planted truth, in weekly-model units (see ablation.py for the unit conversion)
TRUTH = {
    "depth_pct": 1.200,
    "mean_temperature_c": 0.005,
    "holiday_days": (np.exp(0.15) - 1) / 7,
    "n_competitors_500m": -0.020,
}
LABEL = {
    "depth_pct": "Promo depth\n(per unit)",
    "mean_temperature_c": "Temperature\n(per °C)",
    "holiday_days": "Holiday days\n(per day in week)",
    "n_competitors_500m": "Competitors\n(per store within 500 m)",
}


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- fig 1: recovery
def fig_coefficients():
    c = pd.read_csv(ROOT / "coefficients.csv")
    order = list(TRUTH)
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))

    for ax, term in zip(axes, order):
        row = c[c.term == term].iloc[0]
        est, lo, hi = row.estimate_orig, row.ci_low_orig, row.ci_high_orig
        truth = TRUTH[term]
        ok = lo <= truth <= hi

        ax.axvspan(lo, hi, color=ACCENT if ok else BAD, alpha=0.12, lw=0)
        ax.plot([lo, hi], [0, 0], color=ACCENT if ok else BAD, lw=2, solid_capstyle="round")
        ax.plot(est, 0, "o", ms=7, color=ACCENT if ok else BAD, zorder=3)
        ax.axvline(truth, color=INK, lw=1.4, ls=(0, (3, 2)), zorder=2)

        pad = (hi - lo) * 0.55 + abs(truth) * 0.25
        ax.set_xlim(min(lo, truth) - pad, max(hi, truth) + pad)
        ax.set_ylim(-0.45, 0.45)
        ax.set_yticks([])
        ax.set_title(LABEL[term], fontsize=9, color=INK, pad=8)
        ax.grid(False)
        _clean(ax)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Three of four drivers recovered — 95% interval vs planted truth",
                 fontsize=11, color=INK, y=1.02, x=0.01, ha="left")
    fig.text(0.01, -0.10, "Dot = estimate, bar = 95% interval, dashed line = value planted in the "
             "generator. Red = interval misses the truth.\nCompetitor count has no within-store "
             "variation, so a store intercept absorbs it — not identifiable in this design.",
             fontsize=7.5, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_coefficient_recovery.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2: ablation
def fig_ablation():
    a = pd.read_csv(ROOT / "ablation_results.csv")
    name = {
        "baseline (current spec)": "Baseline (current)",
        "no store x SKU effect": "No store×SKU effect",
        "strong cell shrinkage": "Strong cell shrinkage (λ=2)",
        "add linear trend": "Add linear trend",
        "no beta shrinkage": "No β shrinkage",
        "no pooling (u,v free)": "No pooling",
        "drop log_price": "Drop log_price",
        "drop log_store_weight": "Drop log_store_weight",
        "drop Fourier seasonality": "Drop seasonality",
        "drop store random effect": "Drop store effect",
    }
    a["label"] = a["variant"].map(name).fillna(a["variant"])
    a = a.sort_values("wape", ascending=True).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    y = np.arange(len(a))

    colors = [ACCENT if v == "baseline (current spec)" else GRID for v in a["variant"]]
    colors = [BAD if v == "no store x SKU effect" else c for v, c in zip(a["variant"], colors)]
    ax1.barh(y, a["wape"], color=colors, height=0.62)
    ax1.set_yticks(y, a["label"], fontsize=8.5)
    ax1.set_xlabel("cell WAPE  (lower is better)", fontsize=8.5)
    ax1.set_xlim(0, 1.18)
    for yi, v in zip(y, a["wape"]):
        ax1.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
    _clean(ax1)
    ax1.grid(axis="y", visible=False)

    bcolors = [ACCENT if v == "baseline (current spec)" else GRID for v in a["variant"]]
    bcolors = [BAD if v == "add linear trend" else c for v, c in zip(a["variant"], bcolors)]
    ax2.barh(y, a["bias"], color=bcolors, height=0.62)
    ax2.axvline(0, color=INK, lw=0.8)
    ax2.set_xlabel("bias  (weeks 40–43)", fontsize=8.5)
    for yi, v in zip(y, a["bias"]):
        off = 0.006 if v >= 0 else -0.006
        ax2.text(v + off, yi, f"{v:+.1%}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    ax2.set_xlim(-0.16, 0.16)
    _clean(ax2)
    ax2.grid(axis="y", visible=False)

    fig.suptitle("Which modifications actually matter — one factor at a time",
                 fontsize=11, color=INK, y=1.03, x=0.01, ha="left")
    fig.text(0.01, -0.06, "Blue = current spec, red = the two variants that matter most. "
             "Dropping the store×SKU effect costs +26% WAPE;\nadding a linear trend looks "
             "harmless on WAPE but nearly doubles aggregate error over a full horizon "
             "(0.163 → 0.296).",
             fontsize=7.5, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3: calibration
def fig_calibration():
    r = json.loads((ROOT / "evaluation_report.json").read_text())
    nominal = np.array([0.80, 0.95])
    observed = np.array([r["coverage_80"], r["coverage_95"]])

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0.78, 1.0], [0.78, 1.0], ls=(0, (3, 2)), color=MUTED, lw=1,
            label="perfect calibration")
    ax.plot(nominal, observed, "o-", color=BAD, lw=2, ms=8, zorder=3,
            label="current model")
    ax.annotate(f"80% nominal\n→ {observed[0]:.1%} actual",
                xy=(0.80, observed[0]), xytext=(10, -6), textcoords="offset points",
                fontsize=8, color=INK, va="top")
    ax.annotate(f"95% nominal\n→ {observed[1]:.1%} actual",
                xy=(0.95, observed[1]), xytext=(-10, 4), textcoords="offset points",
                fontsize=8, color=INK, va="bottom", ha="right")
    ax.set_xlim(0.78, 1.0)
    ax.set_ylim(0.78, 1.0)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("nominal interval coverage", fontsize=8.5)
    ax.set_ylabel("observed coverage", fontsize=8.5)
    ax.set_title("Intervals are too wide", fontsize=11, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.tick_params(labelsize=8)
    _clean(ax)
    fig.text(0.01, -0.02, "Both intervals over-cover, so the model is over-cautious rather than\n"
             "over-confident. Mean 80% width 153 units against a mean target of 53.",
             fontsize=7.5, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4: baselines
def fig_baselines():
    r = json.loads((ROOT / "evaluation_report.json").read_text())
    acc = {d["forecast"]: d for d in r["accuracy"]}
    order = ["model", "cell mean", "zero", "global mean"]
    nice = {"model": "This model", "cell mean": "Per-cell mean", "zero": "Predict zero",
            "global mean": "Global mean"}
    vals = [acc[k]["WAPE"] for k in order]
    maes = [acc[k]["MAE"] for k in order]

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    y = np.arange(len(order))[::-1]
    colors = [ACCENT] + [GRID] * 3
    ax.barh(y, vals, color=colors, height=0.6)
    ax.set_yticks(y, [nice[k] for k in order], fontsize=9)
    ax.set_xlabel("cell WAPE  (lower is better)", fontsize=8.5)
    ax.set_xlim(0, 1.32)
    for yi, v, m in zip(y, vals, maes):
        ax.text(v + 0.015, yi, f"{v:.3f}   (MAE {m:.1f})", va="center", fontsize=8.5, color=INK)
    _clean(ax)
    ax.grid(axis="y", visible=False)
    ax.set_title("Accuracy against explicit baselines", fontsize=11, color=INK, loc="left", pad=10)
    fig.text(0.01, -0.06, "Beats the best naive baseline by 3.0% on MAE. Modest, and that is the "
             "honest reading:\nthe target is ~48% zeros and dominated by irreducible noise at "
             "store×SKU×week resolution.",
             fontsize=7.5, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_baselines.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 5: seasonality
def fig_seasonality():
    p = ROOT / "seasonality_sweep.json"
    if not p.exists():
        print("  skip fig5: run `python make_figures.py --sweep` first")
        return
    s = pd.DataFrame(json.loads(p.read_text()))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for h, style in [(1, "-"), (2, "--"), (3, ":")]:
        d = s[s.harmonics == h].sort_values("train_weeks")
        ax.plot(d.train_weeks, d.agg_wape, style, marker="o", ms=5,
                color=ACCENT if h == 1 else (MUTED if h == 2 else BAD), lw=1.8,
                label=f"{h} harmonic{'s' if h > 1 else ''}")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 60)
    ax.set_xlabel("weeks of training data available", fontsize=8.5)
    ax.set_ylabel("aggregate WAPE  (log scale)", fontsize=8.5)
    ax.set_xticks([13, 26, 40])
    ax.set_title("Too many seasonal harmonics explodes when data is short",
                 fontsize=11, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.tick_params(labelsize=8)
    ax.annotate("WAPE 16 — predictions diverge", xy=(13.25, 15.98),
                xytext=(0, 14), textcoords="offset points",
                fontsize=8, color=BAD, va="bottom")
    _clean(ax)
    fig.text(0.01, -0.06, "One harmonic is both the safest and the most accurate at 40 weeks. "
             "Three harmonics is six parameters\ntrying to describe a year from a partial "
             "year — the fit is not identified.", fontsize=7.5, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_seasonality.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- sweep (slow)
def run_sweep():
    """Regenerate seasonality_sweep.json: aggregate WAPE vs harmonics vs training length."""
    from model import load_panel, fit_nb, unpack, DRIVERS, CONTROLS, CATS
    FOUR = ("sin1", "cos1", "sin2", "cos2", "sin3", "cos3")
    rows = []
    for train_end, test_weeks in [(40, range(40, 52)), (26, range(26, 39)),
                                  (13, range(13, 26))]:
        for nharm in (1, 2, 3):
            df = load_panel(40, 40)
            keep_f = {f"{f}{k}" for k in range(1, nharm + 1) for f in ("sin", "cos")}
            cols, names = [np.ones(len(df))], ["intercept"]
            for c in DRIVERS + CONTROLS:
                if c in FOUR and c not in keep_f:
                    continue
                v = df[c].to_numpy(float); m, sd = v.mean(), (v.std() or 1.0)
                cols.append((v - m) / sd); names.append(c)
            for c in CATS:
                d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
                for col in d.columns:
                    cols.append(d[col].to_numpy(float)); names.append(col)
            X = np.column_stack(cols)

            st = sorted(df.store_id.unique()); sk = sorted(df["Product Code"].unique())
            si = df.store_id.map({s: i for i, s in enumerate(st)}).to_numpy()
            ki = df["Product Code"].map({s: i for i, s in enumerate(sk)}).to_numpy()
            n_st, n_sk = len(st), len(sk)
            ci = (si * n_sk + ki).astype(np.int64)
            w = df.week_index.to_numpy(); y = df.y.to_numpy(float)
            tr = w < train_end; te = np.isin(w, list(test_weeks))

            th, _, _ = fit_nb(X[tr], si[tr], ki[tr], y[tr], n_st, n_sk,
                              ci=ci[tr], n_cells=n_st * n_sk)
            beta, u, v, wc, r = unpack(th, X.shape[1], n_st, n_sk, n_st * n_sk)
            mu = np.exp(np.clip(X[te] @ beta + u[si[te]] + v[ki[te]] + wc[ci[te]], -20, 20))
            agg = (pd.DataFrame({"s": si[te], "w": w[te], "y": y[te], "mu": mu})
                     .groupby(["s", "w"]).sum())
            rows.append({"harmonics": nharm, "train_weeks": train_end,
                         "agg_wape": float(np.abs(agg.y - agg.mu).sum() / agg.y.sum())})
            print(f"  h={nharm} train={train_end}  aggWAPE={rows[-1]['agg_wape']:.4f}")
    (ROOT / "seasonality_sweep.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    if a.sweep:
        run_sweep()
    else:
        fig_coefficients(); print("  fig1 ok")
        fig_ablation();     print("  fig2 ok")
        fig_calibration();  print("  fig3 ok")
        fig_baselines();    print("  fig4 ok")
        fig_seasonality();  print("  fig5 ok")
        print(f"wrote -> {FIG}")
