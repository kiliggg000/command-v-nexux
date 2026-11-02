"""
holdout.py — does the model work on a store it has never seen?

model.run() fits every store, so it cannot answer that. This script does the same
walk-forward with K stores excluded from every fit. Nothing extra is needed to make
them "unseen": a store with no training rows gets u = 0 and its cells get w = 0 under
the priors, so the forecast is beta + v, which is exactly what a new store would get.

One difference from run(), deliberately: the standardising means and deviations are
fitted on the training rows of each block, not on the whole panel. A holdout whose
scaler has seen the held-out stores is not a holdout. The median fills in load_panel
are still panel-wide; they move the result by very little and they move it in the
held-out stores' favour, so the numbers here are, if anything, flattering.

    python holdout.py --stores 140 --skus 120 --hold 20
    python holdout.py --stores 40 --skus 40 --hold 10 --seed 1   # fast check

Writes holdout_report.json. Does not touch forecast_weekly.parquet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import model as M

ROOT = Path(__file__).resolve().parent


def fit_scaler(df: pd.DataFrame) -> dict:
    """Column means/deviations and categorical levels, from these rows only."""
    num = {c: (float(df[c].mean()), float(df[c].std() or 1.0))
           for c in M.DRIVERS + M.CONTROLS}
    cat = {c: sorted(df[c].astype(str).unique()) for c in M.CATS}
    return {"num": num, "cat": cat}


def build_X(df: pd.DataFrame, sc: dict):
    """Same columns and order as model.design_matrix, but from a supplied scaler."""
    cols, names = [np.ones(len(df))], ["intercept"]
    for c in M.DRIVERS + M.CONTROLS:
        m, s = sc["num"][c]
        cols.append((df[c].to_numpy(float) - m) / s)
        names.append(c)
    for c in M.CATS:
        v = df[c].astype(str).to_numpy()
        for lv in sc["cat"][c][1:]:          # drop_first, as in design_matrix
            cols.append((v == lv).astype(float))
            names.append(f"{c}_{lv}")
    return np.column_stack(cols), names


def wape(y, f):
    return float(np.abs(y - f).sum() / np.abs(y).sum())


def score(d: pd.DataFrame) -> dict:
    y, mu = d["y"].to_numpy(float), d["mu"].to_numpy(float)
    lvl = d.groupby("store_id").apply(
        lambda g: g["mu"].sum() / g["y"].sum(), include_groups=False)
    mu_lvl = mu * d["store_id"].map(lvl).to_numpy(float)
    return {
        "n_rows": int(len(d)),
        "wape": round(wape(y, mu), 4),
        "wape_level_corrected": round(wape(y, mu_lvl), 4),
        "bias": round(float((y - mu).sum() / y.sum()), 4),
        "coverage_80": round(float(((y >= d["lo80"]) & (y <= d["hi80"])).mean()), 4),
        "zero_share": round(float((y == 0).mean()), 4),
    }


def main(stores, skus, hold, seed=0, train_end=40, refit_every=4, last_week=51):
    df = M.load_panel(stores, skus)
    store_ids = sorted(df["store_id"].unique())
    sku_ids = sorted(df["Product Code"].unique())
    si = df["store_id"].map({s: i for i, s in enumerate(store_ids)}).to_numpy()
    ki = df["Product Code"].map({s: i for i, s in enumerate(sku_ids)}).to_numpy()
    y = df["y"].to_numpy(float)
    weeks = df["week_index"].to_numpy()
    n_st, n_sk = len(store_ids), len(sku_ids)
    n_cells = n_st * n_sk
    ci = (si * n_sk + ki).astype(np.int64)

    rng = np.random.default_rng(seed)
    held_ids = sorted(rng.permutation(store_ids)[:hold].tolist())
    held = df["store_id"].isin(held_ids).to_numpy()
    print(f"panel {n_st} stores x {n_sk} SKUs; holding out {len(held_ids)} stores "
          f"(seed {seed}); training on {n_st - len(held_ids)}")

    frames = []
    for start in range(train_end, last_week + 1, refit_every):
        block = list(range(start, min(start + refit_every, last_week + 1)))
        tr = (weeks < start) & ~held
        te = np.isin(weeks, block)
        sc = fit_scaler(df[tr])
        X, names = build_X(df, sc)
        theta, ok, _ = M.fit_nb(X[tr], si[tr], ki[tr], y[tr], n_st, n_sk,
                                ci=ci[tr], n_cells=n_cells)
        beta, u, v, w, r = M.unpack(theta, X.shape[1], n_st, n_sk, n_cells)
        eta = X[te] @ beta + u[si[te]] + v[ki[te]] + w[ci[te]]
        mu = np.exp(np.clip(eta, -20, 20))
        q = M.predict(mu, r)
        frames.append(pd.DataFrame({
            "store_id": df.loc[te, "store_id"].to_numpy(),
            "Product Code": df.loc[te, "Product Code"].to_numpy(),
            "week_index": weeks[te], "held": held[te],
            "y": y[te], "mu": mu,
            "lo80": q[0.1], "hi80": q[0.9],
        }))
        print(f"  weeks {block[0]}-{block[-1]}: fitted, converged={ok}")

    fc = pd.concat(frames, ignore_index=True)
    held_res, seen_res = score(fc[fc.held]), score(fc[~fc.held])

    print(f"\nheld-out stores  WAPE {held_res['wape']:.4f}  "
          f"(level-corrected {held_res['wape_level_corrected']:.4f})  "
          f"bias {held_res['bias']:+.3f}  cov80 {held_res['coverage_80']:.3f}")
    print(f"seen stores      WAPE {seen_res['wape']:.4f}  "
          f"bias {seen_res['bias']:+.3f}  cov80 {seen_res['coverage_80']:.3f}")
    print("predict-zero     WAPE 1.0000")

    out = {
        "purpose": "cold-start check: K stores excluded from every fit",
        "n_stores": n_st, "n_skus": n_sk, "n_held": len(held_ids), "seed": seed,
        "held_store_ids": held_ids,
        "train_end": train_end, "refit_every": refit_every,
        "scored_weeks": [train_end, last_week],
        "preprocessing": "scaler fitted on each block's training rows only",
        "held": held_res, "seen": seen_res,
        "reference": {"predict_zero_wape": 1.0},
    }
    (ROOT / "holdout_report.json").write_text(json.dumps(out, indent=2))
    print("\nwrote holdout_report.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", type=int, default=140)
    ap.add_argument("--skus", type=int, default=120)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-end", type=int, default=40)
    ap.add_argument("--refit-every", type=int, default=4)
    a = ap.parse_args()
    main(a.stores, a.skus, a.hold, a.seed, a.train_end, a.refit_every)
