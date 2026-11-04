# generate_datasets.py
# ---------------------------------------------------------------------------
# CLC2026 synthetic dairy demand -- D1 (pure noise), D2 (real relationships),
# D3 (D2 + planted weak relationships).
#
# Implements: outputs/CLC2026_Data_Generation_Spec.md  (locked decisions, 2026-10-27)
#   * Granularity: generate DAILY, roll up to WEEKLY (decision 1, locked).
#   * D2: log-linear demand surface from REAL HK covariates (store weight, competitor
#         proximity, observed promo depth, calendar, weather) + assumed coefficients.
#   * D1: pure-noise control -- D2's mu is shuffled across ALL cells (destroys every
#         spatial/temporal pattern) but keeps the same carry-probability mask, so its
#         marginal mean AND variance match D2 exactly. This is the noise floor / leak gate.
#   * D3: D2 + a holiday x promo booster (+5%) for stores in districts that are
#         above-median income AND larger-household (REAL census demographics, file 04),
#         plus a spurious covariate (district female share) with ZERO true effect.
#   * Sparsity: ~86.8% zeros at SKU x store x day via a carry-probability mask.
#   * Calibration: beta0 is solved so total annual tonnage hits the target. Temporal
#         betas and target are ASSUMED (MEVGAL is NOT HK-specific, so it cannot
#         calibrate them -- see spec sec. 7). calibration.json, if present, is an
#         optional manual override only.
#
# *** THIS SCRIPT HAS NOT BEEN RUN. Written for review. Run only on explicit go-ahead. ***
# ---------------------------------------------------------------------------
import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEED = 20261027
rng = np.random.default_rng(SEED)

# D4 variant (opt-in, writes its own files): same seed, same mu surface, same NB draws
# -- only the carry mask changes, so it draws assortment conditional on store type.
#   python generate_datasets.py --variant d4
import sys
VARIANT = "d2"
if "--variant" in sys.argv:
    VARIANT = sys.argv[sys.argv.index("--variant") + 1]
HERE = os.path.dirname(os.path.abspath(__file__))


def fpath(name):
    return os.path.join(HERE, name)


# ---------------------------------------------------------------------------
# Config: ASSUMED defaults. calibration.json (if present) may override the
# temporal betas + target tonnage, but is NOT derived from MEVGAL.
# ---------------------------------------------------------------------------
CFG = {
    # --- structural (assumed; cannot be calibrated without real HK sales) ---
    "beta_pack":  -0.30,   # bigger packs -> fewer units
    "beta_store":  1.00,   # bigger-population district -> proportionally more
    "beta_comp":  -0.02,   # more nearby competitors -> slightly fewer sales
    "beta_promo":  1.20,   # 10%-deep promo -> x e^(0.12) ~ +13% expected (recoverable)
    "r_dispersion": 4.0,   # overdispersion; smaller r = bumpier
    # --- temporal (assumed; MEVGAL not HK-specific so not used to fit these) ---
    "beta_wknd":  0.08, "beta_sun": 0.05, "beta_hol": 0.15,
    "beta_season": 0.10, "beta_trend": 0.05,
    "beta_rain": -0.005, "beta_temp": 0.005,
    # --- calibration / sparsity ---
    "target_tonnes": 30000.0,   # placeholder total annual dairy (kg = *1000); assumed
    "carry_lo": 0.02, "carry_hi": 0.30,   # clamp for carry-probability mask
    "d3_boost": 0.05,            # +5% holiday x promo for qualifying districts
}

CAL_PATH = fpath("calibration.json")
if os.path.exists(CAL_PATH):
    with open(CAL_PATH) as f:
        cal = json.load(f)
    for k in ("beta_season", "beta_hol", "beta_trend", "beta_wknd", "beta_sun", "target_tonnes"):
        if k in cal:
            CFG[k] = cal[k]
    print("[generate] loaded calibration.json -> temporal overrides applied")
else:
    print("[generate] calibration.json NOT found -> assumed temporal defaults (MEVGAL is not HK-specific)")

# Per-type staple base rates (relative expected sales; assumed, not fitted)
TYPE_BASE = {
    'Chilled fresh milk': 1.10, 'UHT milk': 0.95, 'Yoghurt': 0.90, 'Yoghurt drinks': 0.70,
    'Cheese': 0.45, 'Butter': 0.35, 'Cream': 0.25, 'Evaporated milk': 0.55,
    'Condensed milk': 0.55, 'Cheese spread': 0.40,
}

# ---------------------------------------------------------------------------
# Load cleaned inputs
# ---------------------------------------------------------------------------
sku = pd.read_csv(fpath("sku_master.csv"))
stores = pd.read_csv(fpath("store_weights_140.csv"))
comp = pd.read_csv(fpath("competitor_features.csv"))
daily = pd.read_csv(fpath("daily_features_2024.csv")).sort_values("date").reset_index(drop=True)
prices = pd.read_csv(fpath("sku_daily_prices_2024.csv"))
census = pd.read_csv(fpath("04_香港18区人口年龄收入及住户资料_2021普查_中文列标题.csv"), encoding="utf-8-sig")

N_SKU, N_STORE, N_DAY = len(sku), len(stores), len(daily)
assert (N_SKU, N_STORE, N_DAY) == (120, 140, 366), (N_SKU, N_STORE, N_DAY)

# store-side (140,)
store_id = stores["store_id"].to_numpy()
district_zh = stores["district_zh"].to_numpy()
store_weight = stores["store_weight"].to_numpy().astype(float)
comp_index = (comp.set_index("store_id").loc[store_id, "n_supermarket_500m"].to_numpy().astype(float)
              + 0.3 * comp.set_index("store_id").loc[store_id, "n_convenience_500m"].to_numpy().astype(float))

# SKU-side (120,)
prod_code = sku["Product Code"].to_numpy()
sku_base = sku["type"].map(TYPE_BASE).to_numpy().astype(float)
pack_kg = sku["pack_kg"].to_numpy().astype(float)

# daily (366,)
iso_wday = daily["iso_weekday"].to_numpy().astype(int)
is_wknd = daily["is_weekend"].to_numpy().astype(int)
is_sun = daily["is_sunday"].to_numpy().astype(int)
is_hol = np.maximum(daily["is_named_holiday"].to_numpy().astype(int),
                    daily["is_general_holiday"].to_numpy().astype(int))
rain = np.nan_to_num(daily["rainfall_mm"].to_numpy().astype(float), nan=0.0)   # 73 missing readings -> 0
temp = np.nan_to_num(daily["mean_temperature_c"].to_numpy().astype(float), nan=0.0)
t = np.arange(N_DAY)
season = np.sin(2 * np.pi * t / 365.25)
trend = t / float(N_DAY)

# depth grid (SKU x day)
depth = (prices.pivot_table(index="Product Code", columns="date", values="depth_pct")
         .reindex(index=prod_code, columns=daily["date"]).fillna(0.0).to_numpy().astype(float))

# REAL census demographics joined to stores (140,)
census = census.set_index("地區名稱（中文）")
hh_size = census.loc[district_zh, "家庭住戶平均人數（人／戶）"].to_numpy().astype(float)
med_income = census.loc[district_zh, "家庭住戶每月收入中位數（港元）"].to_numpy().astype(float)
female = census.loc[district_zh, "女性人口（人）"].to_numpy().astype(float)
male = census.loc[district_zh, "男性人口（人）"].to_numpy().astype(float)
gender_mix = female / (female + male)            # district female share (spurious covariate, zero effect)
qual_store = ((med_income > np.median(med_income)) &
              (hh_size > np.median(hh_size)))     # D3 qualifying districts (140,)

# ---------------------------------------------------------------------------
# D2 log-demand surface  lp = log(mu)  shape (140,120,366)
# ---------------------------------------------------------------------------
r = CFG["r_dispersion"]
lp = (CFG["beta_store"] * np.log(store_weight))[:, None, None] \
   + (CFG["beta_comp"] * comp_index)[:, None, None] \
   + (np.log(sku_base) + CFG["beta_pack"] * np.log(pack_kg))[None, :, None] \
   + (CFG["beta_wknd"] * is_wknd + CFG["beta_sun"] * is_sun + CFG["beta_hol"] * is_hol
      + CFG["beta_season"] * season + CFG["beta_trend"] * trend
      + CFG["beta_rain"] * rain + CFG["beta_temp"] * (temp - 20.0))[None, None, :] \
   + CFG["beta_promo"] * depth[None, :, :]

# beta0 calibration: scale so total EXPECTED kg == target_tonnes * 1000
carry_base = rng.beta(1.0, 6.58, size=(N_STORE, N_SKU))
carry = np.clip(carry_base, CFG["carry_lo"], CFG["carry_hi"])

if VARIANT == "d4":
    # Each store gets a type from the district median-income quartile -- a label the
    # retailer has when the store opens, not something inferred from sales. Each type
    # carries its own core assortment: 55 per cent of SKUs at ~4x the rate of the rest.
    # The draw keeps D2's ranks (probability-integral transform of the same base draw),
    # so the two panels differ only in how assortment is distributed across stores.
    from scipy.stats import beta as beta_dist
    q = np.quantile(med_income, [0.25, 0.50, 0.75])
    store_type = np.digitize(med_income, q)                     # (140,) in 0..3
    core_rng = np.random.default_rng(SEED + 1)
    core = core_rng.random((4, N_SKU)) < 0.55                   # (4,120)
    ratio = np.where(core, 2.20, 0.55)                          # 4:1 core vs tail
    base_mean = 1.0 / 7.58
    target_mean = np.clip(base_mean * ratio / (0.55 * 2.20 + 0.45 * 0.55), 0.02, 0.60)
    kappa = 7.58
    a = target_mean * kappa
    b = (1.0 - target_mean) * kappa
    u_pit = 1.0 - (1.0 - carry_base) ** 6.58                    # CDF of Beta(1, 6.58)
    carry = np.clip(beta_dist.ppf(u_pit, a[store_type], b[store_type]),
                    CFG["carry_lo"], CFG["carry_hi"])
    print(f"[d4] store types {np.bincount(store_type).tolist()}; "
          f"carry mean {carry.mean():.4f} (d2 {carry_base.mean():.4f})")
mu0 = np.exp(lp)
expected_kg = np.sum(carry[:, :, None] * mu0 * pack_kg[None, :, None])
beta0 = np.log(CFG["target_tonnes"] * 1000.0 / expected_kg)
lp += beta0
mu = np.exp(lp)                                   # D2 conditional mean surface

# censoring / active mask (SKU x store, constant across days)
active = rng.random((N_STORE, N_SKU, N_DAY)) < carry[:, :, None]

# Negative-Binomial draw: NB(n=r, p=r/(r+mu)) has mean mu, var mu+mu^2/r
def draw_nb(mean):
    return rng.negative_binomial(r, r / (r + mean))

qty_d2 = np.where(active, draw_nb(mu), 0.0).astype(np.int32)

# ---------------------------------------------------------------------------
# D1 = pure-noise control: shuffle D2's mu across ALL cells, keep same mask.
# Matches D2 marginal mean AND variance exactly; no learnable structure.
# ---------------------------------------------------------------------------
mu1 = mu.reshape(-1).copy(); rng.shuffle(mu1); mu1 = mu1.reshape(mu.shape)
qty_d1 = np.where(active, draw_nb(mu1), 0.0).astype(np.int32)

# ---------------------------------------------------------------------------
# D3 = D2 + planted holiday x promo booster (+5%) for qualifying stores on holidays
# ---------------------------------------------------------------------------
boost = (CFG["d3_boost"] * CFG["beta_promo"] * depth[None, :, :]
         * is_hol[None, None, :] * qual_store[:, None, None])
mu3 = np.exp(lp + boost)
qty_d3 = np.where(active, draw_nb(mu3), 0.0).astype(np.int32)

# ---------------------------------------------------------------------------
# Weekly roll-up (7-day blocks from 2024-01-01). 366 days -> 53 weeks (last partial).
# ---------------------------------------------------------------------------
week_idx = (t // 7).astype(int)
N_WEEK = int(week_idx.max()) + 1
start = date(2024, 1, 1)


def to_weekly(qty):
    # qty is (N_STORE, N_SKU, N_DAY); move day to the front before flattening,
    # otherwise the 7-day blocks mix different stores/SKUs (silent corruption).
    qf = qty.transpose(2, 0, 1).reshape(N_DAY, N_STORE * N_SKU)   # (366, 16800) day-major
    out = np.zeros((N_STORE * N_SKU, N_WEEK))
    np.add.at(out, (np.arange(N_STORE * N_SKU)[:, None], week_idx[None, :]), qf.T)
    return out.reshape(N_STORE, N_SKU, N_WEEK)


w_d1, w_d2, w_d3 = to_weekly(qty_d1), to_weekly(qty_d2), to_weekly(qty_d3)

# ---------------------------------------------------------------------------
# Main weekly table (store x sku x week)
# ---------------------------------------------------------------------------
si, ki, wi = np.meshgrid(np.arange(N_STORE), np.arange(N_SKU), np.arange(N_WEEK), indexing="ij")
si, ki, wi = si.ravel(), ki.ravel(), wi.ravel()
flat = pd.DataFrame({
    "store_id": store_id[si],
    "store_name": stores["store_name"].to_numpy()[si],
    "district_zh": district_zh[si],
    "Product Code": prod_code[ki],
    "Product Name": sku["Product Name"].to_numpy()[ki],
    "Brand": sku["Brand"].to_numpy()[ki],
    "type": sku["type"].to_numpy()[ki],
    "pack_kg": pack_kg[ki],
    "week_index": wi.astype(np.int32),
    "week_start": [(start + timedelta(days=int(k) * 7)).isoformat() for k in wi],
    "qty_d1": w_d1.ravel().astype(np.int32),
    "qty_d2": w_d2.ravel().astype(np.int32),
    "qty_d3": w_d3.ravel().astype(np.int32),
    "spurious_z": np.repeat(gender_mix, N_SKU * N_WEEK)[si],   # zero true effect
})
SUFFIX = "" if VARIANT == "d2" else "_" + VARIANT
flat.to_parquet(fpath(f"dairy_weekly_d1d2d3{SUFFIX}.parquet"), index=False)

# The true weekly conditional mean. Only meaningful where the variant exists: it is the
# best any forecaster could do, so it sets the floor on achievable WAPE for that panel.
if VARIANT != "d2":
    w_mu = to_weekly(mu * carry[:, :, None])   # carry zeroes the days the store does not stock it
    pd.DataFrame({"store_id": store_id[si], "Product Code": prod_code[ki],
                  "week_index": wi.astype(np.int32), "mu_true": w_mu.ravel()}
                 ).to_parquet(fpath(f"weekly_mu{SUFFIX}.parquet"), index=False)

# Demo subset (first 40 SKU x first 60 store x weeks) for easy EDA
demo = flat[(flat["store_id"].isin(store_id[:60])) & (flat["Product Code"].isin(prod_code[:40]))]
demo.to_parquet(fpath(f"dairy_weekly_demo{SUFFIX}.parquet"), index=False)
demo.to_csv(fpath(f"dairy_weekly_demo{SUFFIX}.csv"), index=False)

# Daily recovery subset (first 30 store x first 40 SKU x 366 day) with full daily features
ri, ki2, ti = np.meshgrid(np.arange(30), np.arange(40), np.arange(N_DAY), indexing="ij")
ri, ki2, ti = ri.ravel(), ki2.ravel(), ti.ravel()
pflag = (depth > 0).astype(int)
rec = pd.DataFrame({
    "store_id": store_id[ri],
    "Product Code": prod_code[ki2],
    "type": sku["type"].to_numpy()[ki2],
    "date": daily["date"].to_numpy()[ti],
    "iso_weekday": daily["iso_weekday"].to_numpy()[ti],
    "is_weekend": is_wknd[ti],
    "is_general_holiday": is_hol[ti],
    "rainfall_mm": rain[ti],
    "promo_flag": pflag[ki2, ti],
    "depth_pct": depth[ki2, ti],
    "qty_d1": qty_d1[ri, ki2, ti],
    "qty_d2": qty_d2[ri, ki2, ti],
    "qty_d3": qty_d3[ri, ki2, ti],
    "spurious_z": np.repeat(gender_mix[:30], 40 * N_DAY)[ri],
})
rec.to_parquet(fpath(f"dairy_daily_recovery{SUFFIX}.parquet"), index=False)
rec.to_csv(fpath(f"dairy_daily_recovery{SUFFIX}.csv"), index=False)

# ---------------------------------------------------------------------------
# Validation + run config
# ---------------------------------------------------------------------------
def tonnes(w):
    return float((w * pack_kg[None, :, None]).sum()) / 1000.0

run_cfg = {
    "seed": SEED,
    "n_store": N_STORE, "n_sku": N_SKU, "n_day": N_DAY, "n_week": N_WEEK,
    "cfg": CFG,
    "carry_mean": float(carry.mean()),
    "realized_zero_rate": float((qty_d2 == 0).mean()),
    "d2_total_tonnes": tonnes(w_d2),
    "d1_total_tonnes": tonnes(w_d1),
    "d3_total_tonnes": tonnes(w_d3),
    "noise_floor_check": {
        "d1_mean_vs_d2_mean_ratio": float(w_d1.mean() / w_d2.mean()),
        "d1_var_vs_d2_var_ratio": float(w_d1.var() / w_d2.var()),
    },
    "d3_qualifying_stores": int(qual_store.sum()),
}
with open(fpath(f"generation_run_config{SUFFIX}.json"), "w") as f:
    json.dump(run_cfg, f, indent=2)

print("[generate] script ready (NOT executed). Summary would be:")
print(json.dumps({k: run_cfg[k] for k in
      ("carry_mean", "realized_zero_rate", "d2_total_tonnes", "d1_total_tonnes",
       "d3_total_tonnes", "d3_qualifying_stores")}, indent=2))
print("noise_floor_check:", run_cfg["noise_floor_check"])
