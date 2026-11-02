# Model results — first code pass

Run date: 2026-10-29. Scope: `--stores 40 --skus 40`, train weeks 0–39, test weeks 40–51
(week 52 excluded: 366 days = 52 full weeks + a 2-day stub, which is not a comparable
period). All numbers reproducible from the commands at the bottom.

---

## 1. Four data bugs found and fixed

These are worth stating first, because three of them were *silently* corrupting results —
the pipeline ran to completion and produced plausible-looking numbers throughout.

### 1.1 Weekly roll-up mixed different stores/SKUs (`generate_datasets.py`)

`to_weekly()` reshaped `(140, 120, 366)` as `qty.reshape(366, 16800)` with **no transpose**,
so each 7-day block pulled days from different stores and SKUs. Every weekly figure was a
blend.

- Symptom: daily data showed a strong promo effect (promo weeks 12.04 vs 6.06 units,
  corr 0.354) while the weekly table showed **none** (54.71 vs 54.69, corr 0.021).
- Verification: daily-sum vs weekly rows matched exactly on only **19.96%** of cells.
- Fix: `qty.transpose(2, 0, 1).reshape(N_DAY, N_STORE * N_SKU)` — day-major.
- After: exact match **100.00%**, MAE 0.0, promo uplift 72.8 vs 39.5 units (1.84×).

### 1.2 Weekly promo depth averaged over the wrong denominator (`build_panel.py`)

The price source is **event-based**, not a full daily panel: 37,610 rows for 120 SKUs ×
362 dates (43,440 expected), with as few as 49 observed dates for some SKUs. The generator
treats an absent day as `depth = 0`; the panel averaged depth over only the days that
happened to have rows.

A week with 2 promo days and 5 absent days therefore recorded depth 0.20 instead of
`2 × 0.20 / 7 = 0.057`. That is a multiplicative measurement error on the regressor, which
biases the estimated coefficient toward zero.

- Symptom: within-SKU slope on full-week promos was **−0.08** against a planted **1.20**;
  panel depth matched the generator's daily mean on only 64% of rows.
- Fix: reindex onto the full 366-day calendar first, fill depth/promo with 0, forward-fill
  price, then average over 7 days.
- After: match **100.00%**, MAE 0.0. `depth_pct` coefficient went **0.58 → 1.15**.

### 1.3 Wrong census column for income (`build_panel.py`)

The panel used `每月主要職業收入中位數` (median **occupational** income) while the generator
uses `家庭住戶每月收入中位數` (median **household** income). Two different variables.

- Consequence: the `log_income` control was the wrong covariate, and the D3 qualifying-store
  set was wrong.
- Fix: use the household column. Qualifying stores now come out at **49 of 140**, which
  matches the generator's own `generation_run_config.json` (`d3_qualifying_stores: 49`).

### 1.4 Holiday flags and the final-week stub

`is_hol = max(is_named_holiday, is_general_holiday)`. In HK, **Sunday is a general holiday**,
so `holiday_days ≥ 1` for 98% of weeks — the level is absorbed by the intercept and only the
named holidays provide variation. Additionally week 52 has only 2 days. Both are handled
explicitly (aligned flag definition; incomplete weeks excluded from scoring). Week 52 alone
moved aggregate bias from +3% to +12%.

---

## 2. Model

```
log μ = α + u[store] + v[sku] + w[store×sku] + β'x + Fourier(3 harmonics)
u ~ N(0, σ_u²),  v ~ N(0, σ_v²),  w ~ N(0, σ_w²)      (MAP / Laplace)
y ~ NegBin(μ, r)                                      var = μ + μ²/r
```

Estimated σ: `σ_u = 0.146`, `σ_v = 0.178`, `σ_w = 0.873`, `r = 0.183`.

**The store×SKU term is the single most important addition.** The generator thins each day by
an availability probability `carry ~ Beta(1, 6.58)` (mean ≈ 0.13), so `log(carry)` is a
store×SKU effect. Without it the model inflates dispersion to explain the zeros and cell WAPE
is **1.07**; with it, **0.85** (−21%). No store/SKU-only hierarchy can represent this.

### Estimated driver coefficients vs planted truth

| driver | estimate | 95% CI | planted | covers truth? |
|---|---|---|---|---|
| `depth_pct` | 1.090 | [0.829, 1.351] | 1.200 | ✅ |
| `mean_temperature_c` | 0.00459 | [−0.0064, 0.0155] | 0.005 | ✅ |
| `holiday_days` | 0.01726 | [−0.0112, 0.0457] | 0.0231* | ✅ |
| `n_competitors_500m` | −0.00552 | [−0.0076, −0.0035] | −0.020 | ❌ |

\* **Unit conversion matters.** `depth_pct` and `mean_temperature_c` are *means* over 7 days,
so the daily β passes through unchanged. `holiday_days` is a *count*: the effect applies to
only n of 7 days, so the weekly-equivalent coefficient is `(e^0.15 − 1)/7 = 0.0231`, not 0.15.
Comparing a weekly model against the daily β directly is an apples-to-oranges error.

**`n_competitors_500m` is not identifiable in this design.** It has *zero* within-store
variation, so a store random intercept absorbs it completely. The reported value is a prior
artifact, not an estimate. This is a real limitation to state in the tech doc, not a bug.

---

## 3. Validation results

> **Note:** §3 reports the 40×40 subset used for fast iteration. The full 140×120 run has
> since been done and is better on every accuracy metric — see **§7**. The saved artefacts
> (`forecast_weekly.parquet`, `evaluation_report.json`) now reflect the full run.

### 3.1 Point accuracy vs explicit baselines

Cell level (19,200 rows, 40 stores × 40 SKUs × 12 weeks, 47.9% zeros):

| forecast | WAPE | MAE | bias |
|---|---|---|---|
| **model** | **0.8312** | **49.48** | **+3.6%** |
| predict zero | 1.0000 | 59.77 | +100% |
| global mean | 1.1462 | 68.50 | −4.7% |
| per-cell mean | 0.8458 | 50.55 | −4.7% |

MAE skill vs best naive baseline: **+2.2%**. Modest, and that is the honest reading — at
store×SKU×week resolution the target is ~48% zeros and dominated by irreducible noise.

Store-week aggregate (sum over SKUs): model WAPE **0.1682** vs cell-mean 0.1701.
The model wins on both resolutions, but narrowly at cell level.

### 3.2 Bias

Mean per-store bias **+3.8%** (range −10.1% to +14.1%). Earlier configurations had a
systematic +10–15% under-forecast on *every* store; see §4.

### 3.3 Interval calibration — **fails, over-conservative**

| nominal | observed |
|---|---|
| 80% | 91.3% |
| 95% | 98.5% |

Mean 80% interval width is **173.5 units** against a mean target of 59.8. The analytic
negative-binomial intervals are far too wide. This is expected: `r` absorbs the unmodelled
cell-level heterogeneity, and NB quantiles are discrete and conservative. This is exactly the
case for the conformal / bootstrap comparison in the spec — the analytic interval should not
be shipped as-is.

### 3.4 Recovery tests

| test | criterion | result |
|---|---|---|
| **D1** (pure noise) | promo coefficient must collapse | **−0.049 → PASS** |
| **D2** (real signal) | promo coefficient must appear | **+1.160 → PASS** |
| **D3** (planted boost) | boost must be detectable | **FAIL — below noise floor** |

D1 is the important one: on data with no learnable structure the model reports a promo
elasticity of −0.05 rather than inventing ~1.2. Note the D1 criterion is *coefficient
collapse*, **not** "must lose to the naive baseline" — a pooled model can legitimately beat a
per-cell mean on pure noise, because shrinking 1,600 noisy cell means is better estimation,
not fake discovery.

**D3 is currently unusable as a positive control.** The planted effect is
`0.05 × 1.2 × depth` on holiday∩promo days, which at weekly resolution is ≈0.3–1.4%. Paired
test over 37,632 affected store-weeks: gap **+0.277%** against a minimum detectable effect of
**1.246%** (80% power, two-sided 5%). To make D3 work, `d3_boost` needs to rise from `0.05`
to roughly **0.25**, or the boost must be applied at weekly rather than daily resolution.
This is a decision for the team, not something to change unilaterally.

---

## 4. Ablation — which modifications actually matter

One factor at a time. Two separate yardsticks, because they disagree.

> **Read the windows carefully.** The ablation scores the four weeks 40–43 (fast iteration),
> while §3 scores 40–51. That is why baseline bias is −4.7% here but +3.8% in §3 — different
> horizons, not a contradiction. The `agg WAPE` column is measured on the full 40–51 window,
> because the aggregate error is dominated by level bias and only shows up over a long horizon.

| variant | cell WAPE | bias (wk 40–43) | depth ratio | agg WAPE (wk 40–51) |
|---|---|---|---|---|
| **baseline (current spec)** | 0.8510 | −4.7% | 0.967 | **0.1634** |
| no store×SKU effect | 1.0728 | −7.7% | 0.951 | — |
| strong cell shrinkage (λ_w=2) | 0.8508 | +2.7% | 0.953 | 0.1744 |
| add linear trend | 0.8323 | +4.9% | 0.971 | **0.2957** |
| no β shrinkage | 0.8509 | −4.7% | 0.967 | — |
| no pooling (u,v free) | 0.8511 | −4.8% | **1.004** | — |
| drop `log_price` | 0.8510 | −4.7% | 0.958 | — |
| drop `log_store_weight` | 0.8509 | −4.7% | 0.965 | — |
| drop Fourier seasonality | 0.8570 | −6.9% | 0.984 | — |
| drop store random effect | 1.0771 | −7.4% | 0.961 | — |

What this says:

1. **Store×SKU random effect: decisive.** 1.073 → 0.851. Everything else is noise by comparison.
2. **β shrinkage and pooling do almost nothing** (0.8509–0.8511). The regularisation is not
   doing the work here; the random-effect structure is.
3. **The linear trend is a trap.** It *improves* cell WAPE (0.8323 vs 0.8510) while nearly
   **doubling** aggregate error (0.2957 vs 0.1634) and inflating bias to +25%. With <1 year of
   training data a linear trend and the annual sinusoid are near-collinear; the fit splits them
   arbitrarily and mis-extrapolates. This is the clearest demonstration that **cell-level WAPE
   alone is the wrong metric** — it is insensitive to exactly the error that matters for
   stock planning.
4. **λ_w matters more than it looks.** λ_w=2 vs 0.5 barely moves cell WAPE (0.8508 vs 0.8510)
   but moves aggregate WAPE 0.1744 → 0.1634 and flips bias sign.
5. **`log_price` contributes nothing** (0.8510 vs 0.8510) — it is mechanically collinear with
   discount depth. Keeping it is harmless but it is not earning its place.
6. **`log_store_weight` is the true store-level driver.** Dropping it lets the competitor count
   absorb more of the store-size effect (comp ratio 0.27 → 0.52), which is confounding, not
   recovery.
7. **`holiday_days` and `temperature` trade off against the Fourier basis** — dropping
   seasonality moves holiday to 1.27× and temperature to 0.36× truth. Within a single year the
   holiday calendar is nearly a deterministic function of the date, so the split between
   "seasonality" and "holiday" is not identified. Report these as descriptive, not causal.

---

## 5. Open issues to resolve before this is submission-grade

| # | issue | severity |
|---|---|---|
| 1 | 80% intervals cover 91.3% — analytic NB is too wide; implement conformal and compare | high |
| 2 | D3 boost too small to detect (needs ≈0.25, not 0.05) | high |
| 3 | `n_competitors_500m` unidentifiable with store random effects — decide: drop it, or report it as descriptive only | medium |
| 4 | Cell-level skill over the best naive baseline is only +2.2% | medium |
| 5 | Whole-year run (140 stores × 120 SKUs) not yet done — all numbers here are on the 40×40 subset | medium |
| 6 | PyMC not installed; MAP/Laplace used instead of full posteriors | low |
| 7 | `holiday_days` vs seasonality not separable within one year | low (document) |

---

## 6. Reproduce

```bash
python generate_datasets.py      # writes D1/D2/D3 (weekly + daily recovery)
python build_panel.py            # panel_weekly.parquet
python model.py --stores 40 --skus 40 --refit-every 4 --train-end 40
python evaluate.py --check-noise
python ablation.py --stores 40 --skus 40
```

Outputs: `forecast_weekly.parquet`, `coefficients.csv`, `diagnostics.json`,
`evaluation_report.json`, `ablation_results.csv`, `ablation_results.json`.

---

## 7. Full-year run and follow-up questions

### 7.1 Full 140 × 120 run (supersedes §3)

218,400 forecasts, **9m26s** for the whole walk-forward (~2.4 min per refit).

| | 40×40 subset | full 140×120 |
|---|---|---|
| cell WAPE | 0.8312 | **0.8168** |
| MAE skill vs best naive | +2.2% | **+3.0%** |
| **store-week aggregate WAPE** | 0.1682 | **0.1002** |
| bias | +3.8% | +5.1% |
| `depth_pct` 95% CI | [0.829, 1.351] | **[1.068, 1.213]** |
| σ_store / σ_cell | 0.146 / 0.873 | 0.093 / 0.909 |

> **Baseline definition changed after this table was written.** Every baseline above was
> frozen on weeks < 40 while the model refitted every four weeks, which handed the model
> four extra weeks of data. `evaluate.py` now refreshes the baselines on the model's own
> cadence; on the same information the full-run skill is **+2.65%**, not +3.0%, and the
> per-cell baseline is 0.8391 rather than 0.8417. `--frozen-baseline` reproduces this
> table exactly. The model's own WAPE is unchanged either way.

Two things to take from this:

- The aggregate WAPE **halved** (0.1682 → 0.1002). Summing 120 SKUs instead of 40 averages
  out far more noise, and the aggregate is the number closest to a stock-planning decision.
- The β interval tightened **3.6×** even though every store×SKU cell still has only 53 weeks.
  More cells do not sharpen any individual cell; they sharpen the *shared* coefficients.

Calibration did **not** improve (80% → 91.3%). That confirms it is a method problem, not a
data-size problem — conformal is still the fix.

### 7.2 Year-on-year for the same date is not constructible here

One year of data. `y[t−52]` is defined for **1 of 53 weeks (1.9% coverage)** — it is an empty
column, not a weak feature.

What can be varied is how the calendar is *indexed*. Tested three ways (train wk 0–39,
test wk 40–51, aggregate WAPE):

| seasonal basis | result |
|---|---|
| week-of-year dummies | **0.82** — bias +71%. Weeks 40–51 never appear in training, so those dummies are dropped and the model has *exactly zero* seasonal information where it is needed. |
| month dummies | 0.16 — bias +0.1%, but only because Oct–Dec were dropped too. It is "no seasonality", not a seasonal model, and it lands near the training average by luck. |
| Fourier, 1 harmonic | **0.1623** — defined everywhere. |

**Harmonic count has to match the data.** Aggregate WAPE by training length:

| training weeks | h=1 | h=2 | h=3 |
|---|---|---|---|
| 40 | 0.1623 | 0.1628 | 0.1634 |
| 26 | 0.1736 | 0.1791 | **0.6766** |
| 13 | 0.3242 | 0.4210 | **15.98** |

Three harmonics is six parameters describing a year from a partial year — the fit is not
identified and predictions diverge. **Recommendation: h=1.** The current default is h=3,
which is survivable at 40 weeks and fragile below it.

A YoY lag becomes genuinely useful with 2+ years, with known failure modes: blind to level
shifts (a refitted store still forecasts like its old self), a full year of lag on structural
change, and promo weeks compared against whatever happened to be on promo a year earlier.

### 7.3 "One model for all the variables" is two separate questions

**(a) One model vs one model per store** — settled. A per-store model gets 53 weeks and
nothing else. Hierarchical pooling wins, and the store×SKU random effect is essentially the
whole gain (cell WAPE 1.07 → 0.85).

**(b) Shared coefficients vs per-category coefficients** — not settled. Promo elasticity by
product type gives point estimates 0.914 to 1.415, with 3 of 10 CIs excluding the planted
1.20. But that test used a crude `log(y+1)` OLS estimator whose bias varies with each
category's zero-rate, so most of that spread is probably estimator artefact. A proper test
needs per-category NB fits, which has **not** been run.

Working rule: **share the coefficients, give each group its own intercept** — which is what
the random effects already do. Split a coefficient only when a formal test says it differs,
because splitting multiplies the parameter count by the number of groups while giving each
group 1/G of the data.

---

## 8. Conclusions

1. **The pipeline was producing plausible numbers while three silent data bugs were active.**
   The weekly roll-up mixed stores and SKUs; weekly promo depth was averaged over the wrong
   denominator; the census income column was the wrong variable. Every one of them ran to
   completion without error. Verification against an independent path (daily → weekly
   reconciliation, planted coefficients, generator run config) is what caught them.
2. **Structure beats regularisation.** The store×SKU random effect is worth −21% WAPE;
   β-shrinkage and pooling changes are worth ~0.0001. Effort belongs in the model's
   structure, not in tuning penalty strengths.
3. **A single metric will mislead you.** The linear trend *improves* cell WAPE while nearly
   doubling aggregate error. Cell-level WAPE is insensitive to exactly the error that
   matters for stock planning, so bias and aggregate error must be reported alongside it.
4. **Units of comparison are a real trap.** A weekly model's `holiday_days` coefficient is
   `(e^β−1)/7`, not β. Comparing a weekly estimate to a daily planted parameter produces a
   fake 16× "failure".
5. **Some parameters are not identifiable, and no amount of data fixes it.** A store-constant
   covariate cannot be separated from a store intercept. This should be stated as a design
   limitation, not reported as an estimate.
6. **Two validation components do not yet do their job.** The 80% intervals over-cover by
   11pp, and the D3 positive control sits below the noise floor. Both are known, both have a
   planned fix, and neither should be presented as passing.
7. **The full-year run is strictly better and costs 10 minutes.** There is no reason to keep
   validating on the subset.
