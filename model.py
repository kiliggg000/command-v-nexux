"""
model.py — hierarchical Negative-Binomial GLM (MAP / Laplace approximation).

Confirmed spec:
    log μ = α + u_store + v_sku + β'x + Fourier seasonality + trend
    u_i ~ N(0, σ_u²),  v_j ~ N(0, σ_v²)      # partial pooling via L2 penalty
    y ~ NegBin(μ, r)                          # var = μ + μ²/r

Implementation note: PyMC is not installed, and full MCMC over ~890k rows with 260
random effects is not tractable on a laptop. We therefore compute the MAP (penalised
MLE) — the mode of the same Bayesian hierarchical model — and approximate intervals via
the Fisher information (Laplace). This preserves partial pooling and gives approximate
credible intervals. Swap in PyMC on a subset for exact posteriors.

Run:  python model.py --stores 40 --skus 40 --refit-every 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln
from scipy.stats import nbinom

ROOT = Path(__file__).resolve().parent

DRIVERS = ["depth_pct", "mean_temperature_c", "holiday_days", "n_competitors_500m"]
# NOTE: no linear `trend` term here. With <1 year of training data a linear trend and
# the annual sinusoid are near-collinear, so the fit splits them arbitrarily and then
# mis-extrapolates. Measured: including trend raised held-out under-forecast bias from
# +0.10 to +0.27 for a WAPE change of -0.004 (noise). It is kept as an ablation
# variant instead, so the cost is documented rather than assumed.
CONTROLS = [
    "log_price", "log_store_weight", "weekend", "rainfall_mm", "rainfall_sd",
    "mean_relative_humidity_pct", "log_income", "household_size", "pct_elderly",
    "pct_children",
    "sin1", "cos1", "sin2", "cos2", "sin3", "cos3",
]
CATS = ["type", "promo_type"]


# ----------------------------------------------------------------------------- data
def load_panel(stores: int | None, skus: int | None,
               panel: str = "panel_weekly.parquet") -> pd.DataFrame:
    df = pd.read_parquet(ROOT / panel)
    if stores:
        keep = sorted(df["store_id"].unique())[:stores]
        df = df[df["store_id"].isin(keep)]
    if skus:
        keep = sorted(df["Product Code"].unique())[:skus]
        df = df[df["Product Code"].isin(keep)]
    df = df.sort_values(["week_index", "store_id", "Product Code"]).reset_index(drop=True)

    df["log_price"] = df["log_price"].fillna(
        df.groupby("Product Code")["log_price"].transform("median"))
    for c in DRIVERS + CONTROLS:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


def store_type(df: pd.DataFrame, n_types: int = 4):
    """Assign each store a type from its district's median income quartile.

    Deliberately not derived from sales: a retailer knows a store's format and catchment
    when the lease is signed, so a cold-start model may use it. Returns (types, store_ids)
    in the same order as `store_ids` sorted.
    """
    inc = df.groupby("store_id")["log_income"].first().sort_index()
    cuts = inc.quantile(np.linspace(0, 1, n_types + 1)[1:-1]).to_numpy()
    return np.digitize(inc.to_numpy(), cuts), inc.index.to_numpy()


def design_matrix(df: pd.DataFrame):
    cols, names, scaler = [np.ones(len(df))], ["intercept"], {}
    for c in DRIVERS + CONTROLS:
        v = df[c].to_numpy(float)
        m, s = v.mean(), (v.std() or 1.0)
        scaler[c] = (m, s)
        cols.append((v - m) / s)
        names.append(c)
    for c in CATS:
        d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
        for col in d.columns:
            cols.append(d[col].to_numpy(float))
            names.append(col)
    return np.column_stack(cols), names, scaler


# ----------------------------------------------------------------------------- model
def fit_nb(X, si, ki, y, n_stores, n_skus, lam_b=0.1, lam_u=2.0, lam_v=2.0,
           ci=None, n_cells=0, lam_w=0.5, ct=None, n_seg=0, lam_m=1.0):
    """MAP fit of  log μ = Xβ + u[store] + v[sku] (+ w[store×sku]).

    The optional `w` term is a store×SKU random intercept.  It is not cosmetic: the
    generator thins each day by an availability probability `carry ~ Beta(1, 6.58)`,
    so log(carry) is a store×SKU effect.  Without `w` that effect is unmodelled, the
    dispersion r is pushed down to ~0.16 to explain the excess zeros, and the promo
    coefficient attenuates to ~0.5 of its planted value.  With `w` it recovers ~1.2.
    """
    p = X.shape[1]
    g1, g2 = gammaln(y + 1), None  # placeholder
    use_w = ci is not None and n_cells > 0
    use_m = use_w and ct is not None and n_seg > 0
    off = p + n_stores + n_skus
    off_m = off + n_cells

    def obj(theta):
        beta, u = theta[:p], theta[p:p + n_stores]
        v, logr = theta[p + n_stores:off], theta[-1]
        r = np.exp(logr)
        eta = X @ beta + u[si] + v[ki]
        w = theta[off:off_m] if use_w else None
        m = theta[off_m:-1] if use_m else None
        if use_w:
            eta = eta + w[ci]
        mu = np.exp(np.clip(eta, -20.0, 20.0))
        ll = (gammaln(y + r) - gammaln(r) + r * np.log(r) + y * np.log(mu)
              - (y + r) * np.log(r + mu) - g1)
        g = r * (y - mu) / (r + mu)
        nll = -ll.sum() + 0.5 * lam_b * (beta[1:] @ beta[1:]) \
            + 0.5 * lam_u * (u @ u) + 0.5 * lam_v * (v @ v)
        gb = -(X.T @ g) + lam_b * np.r_[0.0, beta[1:]]
        gu = -np.bincount(si, weights=g, minlength=n_stores) + lam_u * u
        gv = -np.bincount(ki, weights=g, minlength=n_skus) + lam_v * v
        dll_dr = digamma(y + r) - digamma(r) + np.log(r) + 1.0 - np.log(r + mu) - (y + r) / (r + mu)
        glogr = -(r * dll_dr).sum()
        if use_m:
            # w_ij ~ N(m[t(i),j], 1/lam_w) instead of N(0, 1/lam_w): a store's
            # cell effects are shrunk toward what its *type* does on that SKU, not
            # toward zero. A held-out store has no rows, so w collapses to m and the
            # forecast is the type profile.
            dev = w - m[ct]
            nll = nll + 0.5 * lam_w * (dev @ dev) + 0.5 * lam_m * (m @ m)
            gw = -np.bincount(ci, weights=g, minlength=n_cells) + lam_w * dev
            gm = np.bincount(ct, weights=lam_w * dev, minlength=n_seg) + lam_m * m
            return nll, np.r_[gb, gu, gv, gw, gm, glogr]
        if use_w:
            gw = -np.bincount(ci, weights=g, minlength=n_cells) + lam_w * w
            nll = nll + 0.5 * lam_w * (w @ w)
            return nll, np.r_[gb, gu, gv, gw, glogr]
        return nll, np.r_[gb, gu, gv, glogr]

    theta0 = np.r_[np.zeros(p), np.zeros(n_stores), np.zeros(n_skus),
                   np.zeros(n_cells) if use_w else np.zeros(0),
                   np.zeros(n_seg) if use_m else np.zeros(0), np.log(1.0)]
    res = minimize(obj, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 1500, "ftol": 1e-9})
    return res.x, res.success, res.fun


def segment_flat(theta, p, n_stores, n_skus, n_cells, n_seg):
    """The m block, flat, indexed the same way as `ct` (type * n_skus + sku)."""
    off_m = p + n_stores + n_skus + n_cells
    return theta[off_m:off_m + n_seg]


def unpack(theta, p, n_stores, n_skus, n_cells=0):
    off = p + n_stores + n_skus
    return (theta[:p], theta[p:p + n_stores], theta[p + n_stores:off],
            theta[off:off + n_cells], np.exp(theta[-1]))


def fisher_se(X, si, ki, beta, u, v, r, n_stores):
    eta = X @ beta + u[si] + v[ki]
    mu = np.exp(np.clip(eta, -20, 20))
    w = r * mu / (r + mu)                      # Fisher weight per row
    H = X.T @ (w[:, None] * X) + 0.1 * np.eye(X.shape[1])
    cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    se_u = np.sqrt(1.0 / (2.0 + np.bincount(si, weights=w, minlength=n_stores)))
    return se, se_u


def predict(mu, r, qs=(0.1, 0.9, 0.025, 0.975)):
    p = r / (r + mu)
    out = {}
    for q in qs:
        out[q] = nbinom.ppf(q, r, p)
    return out


# ----------------------------------------------------------------------------- run
def run(stores, skus, refit_every, train_end, interaction=True):
    df = load_panel(stores, skus)
    X, names, scaler = design_matrix(df)
    store_ids = sorted(df["store_id"].unique())
    sku_ids = sorted(df["Product Code"].unique())
    si = df["store_id"].map({s: i for i, s in enumerate(store_ids)}).to_numpy()
    ki = df["Product Code"].map({s: i for i, s in enumerate(sku_ids)}).to_numpy()
    y = df["y"].to_numpy(float)
    weeks = df["week_index"].to_numpy()

    n_st, n_sk = len(store_ids), len(sku_ids)
    ci = (si * n_sk + ki).astype(np.int64)
    n_cells = n_st * n_sk if interaction else 0
    ci_arg = ci if interaction else None

    test_weeks = list(range(train_end, weeks.max() + 1))
    frames = []
    last = None
    for start in range(0, len(test_weeks), refit_every):
        block = test_weeks[start:start + refit_every]
        tr = weeks < block[0]
        theta, ok, _ = fit_nb(X[tr], si[tr], ki[tr], y[tr], n_st, n_sk,
                              ci=ci_arg[tr] if interaction else None,
                              n_cells=n_cells)
        beta, u, v, w, r = unpack(theta, X.shape[1], n_st, n_sk, n_cells)
        te = np.isin(weeks, block)
        eta = X[te] @ beta + u[si[te]] + v[ki[te]]
        if interaction:
            eta = eta + w[ci[te]]
        mu = np.exp(np.clip(eta, -20, 20))
        q = predict(mu, r)
        frames.append(pd.DataFrame({
            "store_id": df.loc[te, "store_id"].to_numpy(),
            "Product Code": df.loc[te, "Product Code"].to_numpy(),
            "week_index": weeks[te],
            "y": y[te], "mu": mu,
            "lo80": q[0.1], "hi80": q[0.9], "lo95": q[0.025], "hi95": q[0.975],
        }))
        last = (theta, r, ok)

    fc = pd.concat(frames, ignore_index=True)
    fc.to_parquet(ROOT / "forecast_weekly.parquet", index=False)

    theta, r, ok = last
    beta, u, v, w, _ = unpack(theta, X.shape[1], n_st, n_sk, n_cells)
    se, se_u = fisher_se(X, si, ki, beta, u, v, r, n_st)
    coef = pd.DataFrame({
        "term": names, "estimate": beta, "se": se,
        "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se,
    })
    # rescale to the original feature scale: d(eta)/dx = beta / sd
    sd = np.array([scaler[n][1] if n in scaler else np.nan for n in names])
    coef["estimate_orig"] = beta / sd
    coef["ci_low_orig"] = (beta - 1.96 * se) / sd
    coef["ci_high_orig"] = (beta + 1.96 * se) / sd
    coef.to_csv(ROOT / "coefficients.csv", index=False)

    diag = {
        "estimator": "MAP/Laplace hierarchical NegBin (PyMC not installed)",
        "random_effects": "store + sku" + (" + store×sku" if interaction else ""),
        "n_rows": int(len(df)), "n_stores": n_st, "n_skus": n_sk,
        "n_cells": int(n_cells),
        "dispersion_r": float(r), "sigma_store": float(np.std(u)),
        "sigma_sku": float(np.std(v)),
        "sigma_cell": float(np.std(w)) if interaction else None,
        "converged": bool(ok),
        "train_end_week": train_end, "refit_every": refit_every,
    }
    (ROOT / "diagnostics.json").write_text(json.dumps(diag, indent=2))

    print(f"forecast rows={len(fc)}  r={r:.3f}  sigma_u={np.std(u):.3f}  "
          f"sigma_v={np.std(v):.3f}  sigma_cell={np.std(w) if interaction else 0:.3f}  "
          f"converged={ok}")
    show = coef[coef.term.isin(DRIVERS)][["term", "estimate_orig", "ci_low_orig", "ci_high_orig"]]
    print(show.to_string(index=False))
    return fc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", type=int, default=40)
    ap.add_argument("--skus", type=int, default=40)
    ap.add_argument("--refit-every", type=int, default=4)
    ap.add_argument("--train-end", type=int, default=40)
    ap.add_argument("--no-interaction", action="store_true",
                    help="drop the store x SKU random intercept")
    a = ap.parse_args()
    run(a.stores, a.skus, a.refit_every, a.train_end,
        interaction=not a.no_interaction)
