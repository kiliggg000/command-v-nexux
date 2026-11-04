"""
build_panel.py — assemble the weekly modelling panel.

Joins the generated weekly sales (D1/D2/D3) with:
  - SKU attributes (shelf life, storage, pack type)
  - competitor features (500 m counts, nearest distance)
  - store weights (district population, weight)
  - census demographics (income, household, age, gender) via district_zh
  - weekly weather/calendar aggregates (temp, rain + rain SD, humidity, holidays, weekend)
  - weekly promo/price features (promo flag, depth, effective price, promo type)

Output: panel_weekly.parquet — one row per store x SKU x week, with y = qty_d2
(the product target; qty_d1/d3 kept for validation).

Run:  python build_panel.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def _read(name: str, **kw) -> pd.DataFrame:
    # utf-8-sig strips the BOM that several source CSVs carry
    return pd.read_csv(ROOT / name, encoding="utf-8-sig", **kw)


def load_sales(name: str = "dairy_weekly_d1d2d3.parquet") -> pd.DataFrame:
    df = pd.read_parquet(ROOT / name)
    df["Product Code"] = df["Product Code"].astype(str)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def load_sku() -> pd.DataFrame:
    sku = _read("sku_master.csv", dtype={"Product Code": str})
    keep = ["Product Code", "shelf_life_days", "storage_condition", "pack_type"]
    return sku[keep].drop_duplicates("Product Code")


def load_competitors() -> pd.DataFrame:
    comp = _read("competitor_features.csv")
    comp = comp.rename(columns={
        "n_total_500m": "n_competitors_500m",
        "nearest_competitor_m": "nearest_competitor_distance_m",
    })
    return comp[["store_id", "n_competitors_500m", "nearest_competitor_distance_m"]]


def load_stores() -> pd.DataFrame:
    st = _read("store_weights_140.csv")
    return st[["store_id", "district_pop", "store_weight"]]


def load_census() -> pd.DataFrame:
    cen = _read("04_香港18区人口年龄收入及住户资料_2021普查_中文列标题.csv")
    pop = cen["總人口（人）"]
    out = pd.DataFrame({
        "district_zh": cen["地區名稱（中文）"],
        "total_pop": pop,
        # MUST match the generator's field: "家庭住戶每月收入中位數（港元）"
        # (household monthly income). The occupational-income column
        # "每月主要職業收入中位數" is a DIFFERENT variable and was used by mistake,
        # which silently changed the log_income control and the D3 qualifying set.
        "median_income": cen["家庭住戶每月收入中位數（港元）"],
        "household_size": cen["家庭住戶平均人數（人／戶）"],
        "pct_elderly": cen["65歲及以上人口（人）"] / pop,
        "pct_children": cen["15歲以下人口（人）"] / pop,
        "gender_mix": cen["女性人口（人）"] / pop,
    })
    return out


def load_weather_weekly() -> pd.DataFrame:
    d = _read("daily_features_2024.csv", parse_dates=["date"])
    d["rainfall_mm"] = d["rainfall_mm"].fillna(0.0)  # generator assumption
    # generator uses is_hol = max(named, general) -- match it
    d["is_hol"] = np.maximum(d["is_named_holiday"].astype(int),
                             d["is_general_holiday"].astype(int))
    d["week_start"] = d["date"] - pd.to_timedelta(d["date"].dt.day_of_week, unit="D")
    wk = d.groupby("week_start").agg(
        mean_temperature_c=("mean_temperature_c", "mean"),
        rainfall_mm=("rainfall_mm", "sum"),
        rainfall_sd=("rainfall_mm", "std"),
        mean_relative_humidity_pct=("mean_relative_humidity_pct", "mean"),
        holiday_days=("is_hol", "sum"),
        weekend_share=("is_weekend", "mean"),
    ).reset_index()
    return wk


def load_promo_weekly() -> pd.DataFrame:
    """Weekly promo/price features, averaged over all SEVEN calendar days.

    The price source is EVENT-BASED, not a full daily panel: a (SKU, date) row exists
    only when the SKU was observed that day.  22,123 of 37,610 rows are explicit
    `no_offer` rows (depth NaN -> 0) and a further ~5.8k (SKU, date) cells are absent
    altogether.  generate_datasets.py pivots this file and calls .fillna(0.0), i.e. an
    absent day means "no promo, depth 0".

    If we instead average depth over only the days that happen to have rows, a week
    with 2 promo days and 5 absent days gets depth = 0.20 rather than 2*0.20/7 = 0.057.
    That is a multiplicative measurement error on the regressor, and it biases the
    estimated promo coefficient toward zero (attenuation).  Verified symptom: the
    within-SKU slope on full-week promos came out -0.08 against a planted 1.20.

    So: reindex onto the full calendar first, fill depth/promo with 0, forward-fill the
    price, then average over 7 days.
    """
    p = _read("sku_daily_prices_2024.csv", parse_dates=["date"],
              dtype={"Product Code": str})
    # effective_unit_price is only defined on offer rows; on no-offer rows the
    # regular Price *is* the effective price.
    p["effective_unit_price"] = p["effective_unit_price"].fillna(p["Price"])

    # canonical calendar = the weather file's dates (366 days), same as the generator
    cal = pd.to_datetime(
        _read("daily_features_2024.csv", usecols=["date"], parse_dates=["date"])["date"]
    ).sort_values().unique()

    skus = p["Product Code"].unique()
    full = pd.MultiIndex.from_product([skus, cal], names=["Product Code", "date"])

    p = (p.set_index(["Product Code", "date"])
          .reindex(full)
          .reset_index())
    p["Product Code"] = p["Product Code"].astype(str)

    # absent day => no promo, zero depth
    p["promo_flag"] = p["promo_flag"].fillna(False).astype(bool)
    p["depth_pct"] = p["depth_pct"].fillna(0.0)
    p["promo_type"] = p["promo_type"].fillna("none")
    # price: carry the last observed price forward within each SKU
    p = p.sort_values(["Product Code", "date"])
    p["effective_unit_price"] = p.groupby("Product Code")["effective_unit_price"].ffill()
    p["Price"] = p.groupby("Product Code")["Price"].ffill()

    p["week_start"] = p["date"] - pd.to_timedelta(p["date"].dt.day_of_week, unit="D")

    def _mode(s: pd.Series):
        m = s.dropna()
        m = m[m != "none"]
        return m.mode().iat[0] if not m.empty else "none"

    pr = p.groupby(["Product Code", "week_start"]).agg(
        promo_flag=("promo_flag", "max"),
        depth_pct=("depth_pct", "mean"),          # mean over all 7 days, not just observed
        promo_days=("promo_flag", "sum"),
        effective_unit_price=("effective_unit_price", "mean"),
        promo_type=("promo_type", _mode),
    ).reset_index()
    return pr


def build(sales: str = "dairy_weekly_d1d2d3.parquet") -> pd.DataFrame:
    df = load_sales(sales)
    df = df.merge(load_sku(), on="Product Code", how="left")
    df = df.merge(load_competitors(), on="store_id", how="left")
    df = df.merge(load_stores(), on="store_id", how="left")
    df = df.merge(load_census(), on="district_zh", how="left")
    df = df.merge(load_weather_weekly(), on="week_start", how="left")
    df = df.merge(load_promo_weekly(), on=["Product Code", "week_start"], how="left")

    # derived features
    df["log_price"] = np.log(df["effective_unit_price"].where(df["effective_unit_price"] > 0))
    df["log_income"] = np.log(df["median_income"].where(df["median_income"] > 0))
    # store_weight is the district population weight the generator uses as its dominant
    # store-level driver (beta_store = 1.00 on log weight). Omitting it makes the
    # competitor count absorb only part of the store-size effect.
    df["log_store_weight"] = np.log(df["store_weight"].where(df["store_weight"] > 0))
    df["promo_flag"] = df["promo_flag"].fillna(0).astype(int)
    df["depth_pct"] = df["depth_pct"].fillna(0.0)
    df["promo_type"] = df["promo_type"].fillna("none")
    df["nearest_competitor_distance_m"] = df["nearest_competitor_distance_m"].fillna(5000.0)
    df["weekend"] = df["weekend_share"]
    df["trend"] = df["week_index"].astype(float)

    ang = 2 * np.pi * df["week_index"] / 52.0
    for k in (1, 2, 3):
        df[f"sin{k}"] = np.sin(k * ang)
        df[f"cos{k}"] = np.cos(k * ang)

    df["y"] = df["qty_d2"]  # product target; qty_d1/d3 used by the validation harness
    return df


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "dairy_weekly_d1d2d3.parquet"
    panel = build(src)
    out = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "panel_weekly.parquet")
    panel.to_parquet(out, index=False)

    print(f"wrote {out}  shape={panel.shape}")
    miss = panel.isna().mean().sort_values(ascending=False)
    print("\nTop missing columns:")
    print(miss[miss > 0].head(10).to_string())
    print("\nSample:")
    print(panel[["store_id", "Product Code", "week_index", "y",
                 "mean_temperature_c", "depth_pct", "n_competitors_500m"]].head())
