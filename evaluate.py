"""
evaluate.py — model-agnostic validation harness.

This file is the durable asset of the project.  The forecast model is meant to be
swapped out; this harness is what stays, so that "is the new model better?" has a
mechanical answer instead of an opinion.

It answers four questions, in order of importance:

  1. ACCURACY   — is the point forecast better than doing nothing?  Reported against
                  three explicit baselines, never in isolation.  WAPE alone is
                  meaningless: on this data a WAPE of 0.85 can be *worse* than
                  predicting zero for every cell, because the target is ~49% zeros.
  2. BIAS       — does the model over- or under-forecast specific stores?  Aggregate
                  bias hides offsetting errors, so this is per store.
  3. CALIBRATION— do the stated 80%/95% intervals actually contain 80%/95% of
                  outcomes?  An over-confident model is worse than a wide one.
  4. RECOVERY   — on D1 (no signal) does the model correctly find nothing, and on D3
                  does it find the planted effect?  This is the anti-hallucination
                  test: a model that "finds" structure in D1 is fitting noise.

Run:  python evaluate.py
      python evaluate.py --check-noise     (adds the D1/D3 recovery test; slower)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


# ------------------------------------------------------------------ metric helpers
def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted absolute percentage error = sum|y-yhat| / sum|y|.  Preferred over MAPE
    because MAPE is undefined/infinite on the ~49% zero cells."""
    d = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / d) if d else np.nan


def mae(y, yhat) -> float:
    return float(np.abs(y - yhat).mean())


def bias_ratio(y, yhat) -> float:
    """Positive = under-forecast (actual exceeds forecast)."""
    s = y.sum()
    return float((y - yhat).sum() / s) if s else np.nan


def coverage(y, lo, hi) -> float:
    return float(((y >= lo) & (y <= hi)).mean())


# ------------------------------------------------------------------ baselines
def add_baselines(fc: pd.DataFrame, panel: pd.DataFrame, train_end: int) -> pd.DataFrame:
    """Attach three naive forecasts computed on the SAME training window.

    zero        : predict 0 for everything.  On a 49%-zero target this is a genuinely
                  strong baseline for absolute error, which is why it must be shown.
    global_mean : predict the training-window mean for every cell.
    cell_mean   : predict each (store, SKU)'s own training-window mean — the "each
                  store already knows itself" baseline the brief warns about.
    """
    tr = panel[panel["week_index"] < train_end]
    # restrict history to the modelled store/SKU subset, but keep ALL weeks so the
    # training window is intact
    tr = tr[tr["store_id"].isin(fc["store_id"].unique())
            & tr["Product Code"].isin(fc["Product Code"].unique())]
    g = float(tr["y"].mean())
    cell = tr.groupby(["store_id", "Product Code"])["y"].mean().rename("cell_mean")

    fc = fc.merge(cell, on=["store_id", "Product Code"], how="left")
    fc["cell_mean"] = fc["cell_mean"].fillna(g)      # unseen cell -> global mean
    fc["zero"] = 0.0
    fc["global_mean"] = g
    return fc


def complete_weeks() -> set[int]:
    """Week indices that contain a full 7 days.

    2024 has 366 days = 52 full weeks + a 2-day stub (week 52).  The stub is not a
    comparable period: its target is ~2/7 of a normal week while its features look
    like a normal week, so scoring it injects a large fake error (it alone moved
    aggregate bias from +0.03 to +0.12).
    """
    wx = pd.read_csv(ROOT / "daily_features_2024.csv", usecols=["date"],
                     parse_dates=["date"])
    t = (wx["date"] - wx["date"].min()).dt.days.to_numpy()
    wk = t // 7
    counts = pd.Series(wk).value_counts()
    return set(int(k) for k, v in counts.items() if v == 7)


# ------------------------------------------------------------------ main report
def main(check_noise: bool = False, train_end: int = 40):
    fc = pd.read_parquet(ROOT / "forecast_weekly.parquet")
    panel = pd.read_parquet(ROOT / "panel_weekly.parquet")
    # NOTE: use the FULL panel here. Filtering it to the forecast's keys would also
    # drop the training weeks, leaving the baselines with no history to average over.
    fc = add_baselines(fc, panel, train_end)

    keep = complete_weeks()
    n_all = len(fc)
    fc = fc[fc["week_index"].isin(keep)].copy()
    print(f"scored {len(fc):,} of {n_all:,} rows "
          f"(dropped {n_all-len(fc):,} rows in incomplete weeks: "
          f"{sorted(set(range(int(fc.week_index.min()), int(fc.week_index.max())+1)) - keep)})")

    y = fc["y"].to_numpy(float)
    print(f"test rows: {len(fc):,}   weeks {int(fc.week_index.min())}-{int(fc.week_index.max())}"
          f"   stores {fc.store_id.nunique()}   SKUs {fc['Product Code'].nunique()}")
    print(f"target: mean {y.mean():.2f}  median {np.median(y):.1f}  zero-rate {(y==0).mean():.1%}")

    # ---------------- 1. accuracy vs baselines ----------------
    print("\n=== 1. POINT ACCURACY (cell level, all test rows) ===")
    rows = []
    for name, col in [("model", "mu"), ("zero", "zero"),
                      ("global mean", "global_mean"), ("cell mean", "cell_mean")]:
        p = fc[col].to_numpy(float)
        rows.append({"forecast": name, "WAPE": round(wape(y, p), 4),
                     "MAE": round(mae(y, p), 3), "bias": round(bias_ratio(y, p), 4)})
    acc = pd.DataFrame(rows)
    print(acc.to_string(index=False))

    best_naive = acc[acc.forecast != "model"]["MAE"].min()
    model_mae = acc.loc[acc.forecast == "model", "MAE"].iloc[0]
    skill = 1 - model_mae / best_naive
    print(f"\nMAE skill vs best naive baseline: {skill:+.1%}"
          f"   ({'BEATS' if skill > 0 else 'LOSES TO'} the best baseline)")

    # ---------------- 2. store-level aggregate + bias ----------------
    print("\n=== 2. STORE-WEEK AGGREGATE (sum over SKUs) ===")
    ag = (fc.groupby(["store_id", "week_index"], as_index=False)
            .agg(y=("y", "sum"), mu=("mu", "sum"),
                 cell_mean=("cell_mean", "sum"), zero=("zero", "sum")))
    print(f"aggregate WAPE  model {wape(ag.y, ag.mu):.4f}   "
          f"cell-mean {wape(ag.y, ag.cell_mean):.4f}")
    print(f"aggregate MAE   model {mae(ag.y, ag.mu):.2f}   "
          f"cell-mean {mae(ag.y, ag.cell_mean):.2f}")

    sb = (fc.groupby("store_id")
            .apply(lambda d: bias_ratio(d["y"].to_numpy(float), d["mu"].to_numpy(float)),
                   include_groups=False)
            .rename("bias").reset_index())
    print(f"\nper-store bias: mean {sb.bias.mean():+.3f}  "
          f"min {sb.bias.min():+.3f}  max {sb.bias.max():+.3f}")
    worst = sb.reindex(sb.bias.abs().sort_values(ascending=False).index).head(5)
    print("worst 5 stores by |bias|:")
    print(worst.to_string(index=False))

    # ---------------- 3. interval calibration ----------------
    print("\n=== 3. INTERVAL CALIBRATION ===")
    cov80 = coverage(y, fc["lo80"].to_numpy(float), fc["hi80"].to_numpy(float))
    cov95 = coverage(y, fc["lo95"].to_numpy(float), fc["hi95"].to_numpy(float))
    print(f"nominal 80% -> observed {cov80:.1%}   (gap {cov80-0.80:+.1%})")
    print(f"nominal 95% -> observed {cov95:.1%}   (gap {cov95-0.95:+.1%})")
    verdict = ("well calibrated" if abs(cov80-0.80) < 0.05 and abs(cov95-0.95) < 0.03
               else "OVER-CONFIDENT (intervals too narrow)" if cov80 < 0.80
               else "OVER-CONSERVATIVE (intervals too wide)")
    print(f"verdict: {verdict}")

    # width vs target scale
    width = (fc["hi80"] - fc["lo80"]).mean()
    print(f"mean 80% interval width: {width:.1f} units vs mean target {y.mean():.1f}")

    # ---------------- 4. optional D1 / D3 recovery ----------------
    noise = None
    if check_noise:
        noise = check_recovery(train_end)

    out = {
        "n_test_rows": int(len(fc)),
        "accuracy": acc.to_dict("records"),
        "mae_skill_vs_best_naive": round(float(skill), 4),
        "aggregate_wape_model": round(wape(ag.y, ag.mu), 4),
        "aggregate_wape_cell_mean": round(wape(ag.y, ag.cell_mean), 4),
        "coverage_80": round(cov80, 4),
        "coverage_95": round(cov95, 4),
        "calibration_verdict": verdict,
        "store_bias_mean": round(float(sb.bias.mean()), 4),
        "store_bias_absmax": round(float(sb.bias.abs().max()), 4),
        "recovery": noise,
    }
    (ROOT / "evaluation_report.json").write_text(json.dumps(out, indent=2))
    print("\nwrote evaluation_report.json")
    return out


# ------------------------------------------------------------------ D1 / D3 test
def check_recovery(train_end: int = 40):
    """Fit the same specification on D1 (pure noise) and D3 (planted boost).

    D1: y is D2's mu shuffled across cells, so there is NO learnable structure.  The
        right test is NOT "does it lose to the naive baseline" — a pooled model can
        legitimately beat a per-cell mean on pure noise, because shrinking 1,600 noisy
        cell means toward a global level is better estimation, not fake discovery.
        The real anti-hallucination test is whether the DRIVER COEFFICIENTS COLLAPSE.
        If the model reports a promo elasticity of ~1.2 on data with no promo signal,
        every D2 result is worthless.

    D3: a +5% holiday x promo boost is planted for high-income/high-household-size
        districts.  A correct model should leave a small POSITIVE residual on exactly
        those store-weeks, and none elsewhere.
    """
    from model import load_panel, design_matrix, fit_nb, unpack

    df = load_panel(40, 40)
    X, names, scaler = design_matrix(df)
    store_ids = sorted(df["store_id"].unique())
    sku_ids = sorted(df["Product Code"].unique())
    si = df["store_id"].map({s: i for i, s in enumerate(store_ids)}).to_numpy()
    ki = df["Product Code"].map({s: i for i, s in enumerate(sku_ids)}).to_numpy()
    n_st, n_sk = len(store_ids), len(sku_ids)
    ci = (si * n_sk + ki).astype(np.int64)
    weeks = df["week_index"].to_numpy()

    # D3 qualifying stores: high income AND high household size (per the generator)
    st = df.groupby("store_id").agg(inc=("log_income", "first"),
                                    hh=("household_size", "first"))
    qual = ((st["inc"] > st["inc"].median()) & (st["hh"] > st["hh"].median()))
    qual = df["store_id"].map(qual).to_numpy(bool)

    tr = weeks < train_end
    te = (weeks >= train_end) & (weeks <= 51)      # exclude the 2-day stub
    res = {}

    for label, col in [("d1_noise", "qty_d1"), ("d2_signal", "qty_d2"), ("d3_planted", "qty_d3")]:
        y = df[col].to_numpy(float)
        theta, ok, _ = fit_nb(X[tr], si[tr], ki[tr], y[tr], n_st, n_sk,
                              ci=ci[tr], n_cells=n_st * n_sk)
        beta, u, v, w, r = unpack(theta, X.shape[1], n_st, n_sk, n_st * n_sk)
        eta = X[te] @ beta + u[si[te]] + v[ki[te]] + w[ci[te]]
        mu = np.exp(np.clip(eta, -20, 20))
        yt = y[te]
        depth = float(beta[names.index("depth_pct")] / scaler["depth_pct"][1])

        # relative residual, positive = actual above forecast
        rel = (yt - mu) / np.clip(mu, 1e-6, None)
        aff = (qual[te] & (df["promo_flag"].to_numpy()[te] == 1)
               & (df["holiday_days"].to_numpy()[te] >= 2))
        res[label] = {
            "wape_model": round(wape(yt, mu), 4),
            "depth_coef": round(depth, 4),
            "resid_affected": round(float(rel[aff].mean()), 4) if aff.any() else None,
            "resid_other": round(float(rel[~aff].mean()), 4),
            "n_affected": int(aff.sum()),
        }
        print(f"  {label}: depth_coef {depth:+.4f}   WAPE {wape(yt, mu):.4f}   "
              f"resid(affected) {res[label]['resid_affected']}  "
              f"resid(other) {res[label]['resid_other']}")

    d1, d2 = res["d1_noise"], res["d2_signal"]
    collapse = abs(d1["depth_coef"]) < 0.25
    detects = abs(d2["depth_coef"]) > 0.8
    print(f"\n  D1 test: depth coef {d1['depth_coef']:+.4f} on pure noise "
          f"-> {'PASS (collapses to zero)' if collapse else 'FAIL (invented structure)'}")
    print(f"  D2 test: depth coef {d2['depth_coef']:+.4f} on real signal "
          f"-> {'PASS (detects effect)' if detects else 'FAIL (missed effect)'}")

    res["verdict"] = {"d1_collapse": bool(collapse), "d2_detect": bool(detects)}
    res["d3_direct"] = d3_direct_check()
    return res


def d3_direct_check():
    """Verify the D3 planted effect model-free.

    D3 = D2 with a +5% x beta_promo x depth boost on holiday days in qualifying
    districts.  Rather than infer it from model residuals (which are ill-conditioned
    when mu is near zero), compare qty_d3 against qty_d2 directly in the daily file,
    where both are present per cell-day.  Expected ratio on affected days:
    exp(0.05 * 1.2 * mean_depth) ~ 1.014; on unaffected days exactly 1.000.
    """
    d = pd.read_parquet(ROOT / "dairy_weekly_d1d2d3.parquet")
    panel = pd.read_parquet(ROOT / "panel_weekly.parquet")
    feats = (panel[["store_id", "Product Code", "week_index", "promo_flag", "holiday_days"]]
             .drop_duplicates(["store_id", "Product Code", "week_index"]))
    d = d.merge(feats, on=["store_id", "Product Code", "week_index"], how="left")

    # qualifying stores: high household income AND high household size, per the generator
    st = panel.groupby("store_id").agg(inc=("median_income", "first"),
                                      hh=("household_size", "first"))
    qual = (st["inc"] > st["inc"].median()) & (st["hh"] > st["hh"].median())
    d["qual"] = d["store_id"].map(qual).fillna(False).to_numpy(bool)

    affected = d["qual"] & (d["promo_flag"] == 1) & (d["holiday_days"] >= 2)
    control = (~d["qual"]) & (d["promo_flag"] == 1) & (d["holiday_days"] >= 2)
    n = int(affected.sum())
    if n == 0:
        print("  D3 test (model-free): no affected store-weeks found -> INCONCLUSIVE")
        return {"n_affected": 0}

    def paired(mask):
        s = d[mask]
        diff = (s["qty_d3"] - s["qty_d2"]).to_numpy(float)
        m = diff.mean()
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        return m, se, len(diff), m / s["qty_d2"].mean()

    m_a, se_a, n_a, rel_a = paired(affected)
    m_c, se_c, n_c, rel_c = paired(control)
    gap, gap_se = rel_a - rel_c, np.sqrt(se_a ** 2 + se_c ** 2) / d.loc[affected, "qty_d2"].mean()

    # minimum detectable effect at 80% power, two-sided 5%
    mde = 2.8 * se_a / d.loc[affected, "qty_d2"].mean()
    detected = abs(gap) > 1.96 * gap_se

    print(f"  D3 test (paired): qualifying stores {int(qual.sum())} of {len(qual)}")
    print(f"    affected {n_a:,} store-weeks: qty_d3-qty_d2 = {m_a:+.3f} "
          f"(t={m_a/se_a:+.2f}, rel {rel_a:+.3%})")
    print(f"    control  {n_c:,} store-weeks: qty_d3-qty_d2 = {m_c:+.3f} "
          f"(t={m_c/se_c:+.2f}, rel {rel_c:+.3%})")
    print(f"    gap {gap:+.3%} vs minimum detectable {mde:.3%} "
          f"-> {'DETECTED' if detected else 'BELOW NOISE FLOOR (D3 is not a usable positive control at this boost size)'}")
    return {"n_qualifying_stores": int(qual.sum()), "n_affected": n_a,
            "rel_affected": round(float(rel_a), 5), "rel_control": round(float(rel_c), 5),
            "gap": round(float(gap), 5), "min_detectable_effect": round(float(mde), 5),
            "detected": bool(detected)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-noise", action="store_true")
    ap.add_argument("--train-end", type=int, default=40)
    a = ap.parse_args()
    main(a.check_noise, a.train_end)
