"""
ablation.py — one-factor-at-a-time ablation on the hierarchical NegBin model.

Purpose (tech doc §4.2d): for each "modification" proposed to the model, does it make
a measurable difference?  We answer with two separate yardsticks, because they are
*not* the same question:

  (1) RECOVERY  — does the estimate land near the value planted in the DGP?
  (2) FORECAST  — does held-out WAPE improve?

A factor can improve one and not the other.  That distinction is the point of the
exercise: a model can forecast well with wrong coefficients (confounded but
predictive), and can recover coefficients while forecasting badly (right structure,
poor variance).  Reporting only one of the two hides this.

Truth comes from generate_datasets.py CFG, not from this file, so the comparison
cannot drift silently:

    beta_promo  = 1.20   (per unit of depth_pct, i.e. 10% deep -> e^0.12 = +12.7%)
    beta_hol    = 0.15   (per holiday day)
    beta_temp   = 0.005  (per degree C above 20)
    beta_comp   = -0.02  (per additional competitor within 500 m)

Run:  python ablation.py --stores 40 --skus 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import CATS, DRIVERS, CONTROLS, fit_nb, unpack, load_panel

ROOT = Path(__file__).resolve().parent

# Planted truth — mirrors generate_datasets.py CFG, CONVERTED to weekly-model units.
#
# This conversion is essential and easy to get wrong.  The generator works at DAILY
# resolution; the panel is weekly.  Whether the daily beta carries through unchanged
# depends on how the feature is aggregated:
#
#   depth_pct          = MEAN of 7 daily depths.  The effect is exp(beta*d) applied
#                        daily, so the weekly sum scales as exp(beta*mean(d)).  The
#                        daily beta carries through UNCHANGED -> 1.20.
#   mean_temperature_c = MEAN of 7 daily temps -> carries through unchanged -> 0.005.
#   holiday_days       = COUNT of holiday days.  The effect applies to only n of the 7
#                        days, so the weekly sum is base*(7 - n + n*e^beta).  Hence
#                        d log(sum)/dn = (e^beta - 1)/7, NOT beta.
#                        (e^0.15 - 1)/7 = 0.0231.
#   n_competitors_500m = store-constant -> unchanged -> -0.02.
BETA_HOL_DAILY = 0.15
TRUTH = {
    "depth_pct": 1.20,
    "holiday_days": (np.exp(BETA_HOL_DAILY) - 1.0) / 7.0,
    "mean_temperature_c": 0.005,
    "n_competitors_500m": -0.02,
}

FOURIER = ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3"]
STORE_LEVEL = ["n_competitors_500m", "log_income", "household_size", "pct_elderly", "pct_children"]


def build_design(df, drivers, controls, cats, drop_terms=(), extra=()):
    """Same scaling convention as model.design_matrix, but with a droppable column set."""
    cols, names, scaler = [np.ones(len(df))], ["intercept"], {}
    for c in list(drivers) + list(controls) + list(extra):
        if c in drop_terms:
            continue
        v = df[c].to_numpy(float)
        m, s = v.mean(), (v.std() or 1.0)
        scaler[c] = (m, s)
        cols.append((v - m) / s)
        names.append(c)
    for c in cats:
        d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
        for col in d.columns:
            cols.append(d[col].to_numpy(float))
            names.append(col)
    return np.column_stack(cols), names, scaler


def wape(y, yhat):
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom else np.nan


def evaluate_variant(X, sd, si, ki, y, weeks, names_keep, lam_b, lam_u, lam_v,
                     use_store_re, use_sku_re, use_cell_re=False, lam_w=0.5,
                     train_end=40, test_weeks=(40, 41, 42, 43)):
    """Fit on weeks < train_end, report driver recovery + held-out WAPE."""
    tr = weeks < train_end
    te = np.isin(weeks, test_weeks)

    n_st = len(np.unique(si)) if use_store_re else 1
    n_sk = len(np.unique(ki)) if use_sku_re else 1
    si_f = si if use_store_re else np.zeros_like(si)
    ki_f = ki if use_sku_re else np.zeros_like(ki)

    # cell index must be built from the *effective* indices, otherwise a variant that
    # drops the store effect still indexes w by the raw store id and overruns n_cells
    ci = (si_f * n_sk + ki_f).astype(np.int64) if use_cell_re else None
    n_cells = n_st * n_sk if use_cell_re else 0

    theta, ok, _ = fit_nb(X[tr], si_f[tr], ki_f[tr], y[tr], n_st, n_sk,
                          lam_b=lam_b, lam_u=lam_u if use_store_re else 1e-8,
                          lam_v=lam_v if use_sku_re else 1e-8,
                          ci=ci[tr] if use_cell_re else None, n_cells=n_cells,
                          lam_w=lam_w)
    beta, u, v, w, r = unpack(theta, X.shape[1], n_st, n_sk, n_cells)

    # X columns are (v - mean)/sd, so d(eta)/dv = beta_i / sd_i
    est = {}
    for i, n in enumerate(names_keep):
        if n in TRUTH:
            est[n] = float(beta[i] / sd[n])
    eta = X[te] @ beta + u[si_f[te]] + v[ki_f[te]]
    if use_cell_re:
        eta = eta + w[ci[te]]
    mu = np.exp(np.clip(eta, -20, 20))
    yt = y[te]
    b = float((yt - mu).sum() / yt.sum()) if yt.sum() else np.nan
    return est, wape(yt, mu), ok, r, b


def main(stores, skus):
    df = load_panel(stores, skus)
    y = df["y"].to_numpy(float)
    weeks = df["week_index"].to_numpy()
    store_ids = sorted(df["store_id"].unique())
    sku_ids = sorted(df["Product Code"].unique())
    si = df["store_id"].map({s: i for i, s in enumerate(store_ids)}).to_numpy()
    ki = df["Product Code"].map({s: i for i, s in enumerate(sku_ids)}).to_numpy()

    # Reference design = current model.py spec
    X0, names0, scaler0 = build_design(df, DRIVERS, CONTROLS, CATS)
    sd0 = {n: scaler0[n][1] for n in scaler0}

    variants = {
        "baseline (current spec)": dict(drop=(), extra=(), lam_b=0.1, lam_u=2.0, lam_v=2.0,
                                        store_re=True, sku_re=True, cell_re=True),
        "no store x SKU effect":   dict(drop=(), extra=(), lam_b=0.1, lam_u=2.0, lam_v=2.0,
                                        store_re=True, sku_re=True, cell_re=False),
        "strong cell shrinkage":   dict(drop=(), extra=(), lam_b=0.1, lam_u=2.0, lam_v=2.0,
                                        store_re=True, sku_re=True, cell_re=True, lam_w=2.0),
        "add linear trend":        dict(drop=(), extra=("trend",), lam_b=0.1, lam_u=2.0,
                                        lam_v=2.0, store_re=True, sku_re=True, cell_re=True),
        "no beta shrinkage":       dict(drop=(), extra=(), lam_b=1e-6, lam_u=2.0, lam_v=2.0,
                                        store_re=True, sku_re=True, cell_re=True),
        "no pooling (u,v free)":   dict(drop=(), extra=(), lam_b=0.1, lam_u=1e-6, lam_v=1e-6,
                                        store_re=True, sku_re=True, cell_re=True),
        "drop log_price":          dict(drop=("log_price",), extra=(), lam_b=0.1, lam_u=2.0,
                                        lam_v=2.0, store_re=True, sku_re=True, cell_re=True),
        "drop log_store_weight":   dict(drop=("log_store_weight",), extra=(), lam_b=0.1,
                                        lam_u=2.0, lam_v=2.0, store_re=True, sku_re=True,
                                        cell_re=True),
        "drop Fourier seasonality": dict(drop=tuple(FOURIER), extra=(), lam_b=0.1, lam_u=2.0,
                                         lam_v=2.0, store_re=True, sku_re=True, cell_re=True),
        "drop store random effect": dict(drop=(), extra=(), lam_b=0.1, lam_u=2.0, lam_v=2.0,
                                         store_re=False, sku_re=True, cell_re=True),
    }

    rows = []
    for label, v in variants.items():
        X, names, scaler = build_design(df, DRIVERS, CONTROLS, CATS,
                                        drop_terms=v["drop"], extra=v["extra"])
        sd = {n: scaler[n][1] for n in scaler}
        est, w, ok, r, b = evaluate_variant(
            X, sd, si, ki, y, weeks, names, v["lam_b"], v["lam_u"], v["lam_v"],
            v["store_re"], v["sku_re"], v["cell_re"], v.get("lam_w", 0.5))
        row = {"variant": label, "wape": round(w, 4), "bias": round(b, 4),
               "r": round(float(r), 4), "converged": bool(ok)}
        for k, t in TRUTH.items():
            row[k] = round(est.get(k, np.nan), 5)
            row[k + "_ratio"] = round(est.get(k, np.nan) / t, 3) if t else np.nan
        rows.append(row)
        print(f"  fitted: {label}")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "ablation_results.csv", index=False)

    print("\n=== coefficient recovery (planted value in parentheses) ===")
    disp = out[["variant", "wape", "bias", "r"] + list(TRUTH)].copy()
    disp.columns = (["variant", "wape", "bias", "disp_r"]
                    + [f"{k} ({t})" for k, t in TRUTH.items()])
    print(disp.to_string(index=False))
    print("\n=== recovery ratio: estimate / planted (1.00 = perfect) ===")
    print(out[["variant"] + [k + "_ratio" for k in TRUTH]].to_string(index=False))

    (ROOT / "ablation_results.json").write_text(json.dumps(
        {"truth": TRUTH, "n_rows": int(len(df)), "rows": rows}, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", type=int, default=40)
    ap.add_argument("--skus", type=int, default=40)
    a = ap.parse_args()
    main(a.stores, a.skus)
