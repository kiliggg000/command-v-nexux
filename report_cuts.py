"""Two cuts of forecast_weekly.parquet used in Section 6.4 of the technical document.

Neither is printed by evaluate.py, so they are kept here to make the numbers reproducible.

    python report_cuts.py

Writes cuts.json and prints both tables.

1. By forecast week. The walk-forward refits every four weeks, so weeks 40-51 fall into
   three blocks (40-43, 44-47, 48-51). "Horizon" is the position inside its block, 1..4.
   This is the check that a four-week block is defensible for a weekly decision.

2. By cell volume. Cells are ranked by total units sold over weeks 40-51 and split into
   quintiles; bias is (sum actual - sum forecast) / sum actual within each quintile.
   This is what shows the estate mean of +5.0 per cent is the net of two opposite errors.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from evaluate import wape  # same definition as the harness

TRAIN_END = 40
LAST_SCORED = 51
REFIT_EVERY = 4


def by_horizon(fc: pd.DataFrame) -> pd.DataFrame:
    f = fc[(fc.week_index >= TRAIN_END) & (fc.week_index <= LAST_SCORED)].copy()
    f["block"] = (f.week_index - TRAIN_END) // REFIT_EVERY
    f["horizon"] = f.week_index - (TRAIN_END + f.block * REFIT_EVERY) + 1

    agg = f.groupby(["horizon", "store_id", "week_index"], as_index=False).agg(
        y=("y", "sum"), mu=("mu", "sum"))

    rows = []
    for h, d in f.groupby("horizon"):
        a = agg[agg.horizon == h]
        rows.append({
            "horizon": int(h),
            "n_rows": int(len(d)),
            "cell_wape": round(wape(d.y.to_numpy(float), d.mu.to_numpy(float)), 4),
            "store_week_wape": round(wape(a.y.to_numpy(float), a.mu.to_numpy(float)), 4),
            "bias": round(float((d.y.sum() - d.mu.sum()) / d.y.sum()), 4),
        })
    return pd.DataFrame(rows)


def by_volume(fc: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    f = fc[(fc.week_index >= TRAIN_END) & (fc.week_index <= LAST_SCORED)]
    g = (f.groupby(["store_id", "Product Code"], as_index=False)
           .agg(y=("y", "sum"), mu=("mu", "sum")))
    g = g[g.y > 0]
    g["quintile"] = pd.qcut(g.y, q, labels=False)
    g["bias"] = (g.y - g.mu) / g.y
    out = (g.groupby("quintile", as_index=False)
             .agg(mean_units=("y", "mean"), mean_bias=("bias", "mean"), n_cells=("bias", "size")))
    out["mean_units"] = out.mean_units.round(0)
    out["mean_bias"] = out.mean_bias.round(4)
    corr = float(np.corrcoef(np.log(g.y), g.bias)[0, 1])
    out.attrs["corr_log_volume_bias"] = round(corr, 3)
    return out


def main() -> None:
    fc = pd.read_parquet("forecast_weekly.parquet")

    h = by_horizon(fc)
    v = by_volume(fc)

    print("\n=== by forecast week (weeks 40-51) ===")
    print(h.to_string(index=False))
    print(f"\nweek 4 vs week 1, cell WAPE: {h.cell_wape.iloc[-1] / h.cell_wape.iloc[0] - 1:+.1%}")
    print(f"week 4 vs week 1, store-week WAPE: "
          f"{h.store_week_wape.iloc[-1] / h.store_week_wape.iloc[0] - 1:+.1%}")

    print("\n=== by cell volume (quintiles, weeks 40-51) ===")
    print(v.to_string(index=False))
    print(f"\ncorr(log volume, bias): {v.attrs['corr_log_volume_bias']:+.3f}")

    json.dump(
        {"by_horizon": h.to_dict("records"),
         "by_volume": v.to_dict("records"),
         "corr_log_volume_bias": v.attrs["corr_log_volume_bias"]},
        open("cuts.json", "w"), indent=2)
    print("\nwrote cuts.json")


if __name__ == "__main__":
    main()
