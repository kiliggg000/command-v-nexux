# CLC2026 — Dairy Sales Data Plan

*Working doc. Scope locked 2026-10-26/27. Updated 2026-10-27.*

**One sentence:** we simulate three dairy-demand datasets on the same store/SKU/date grid so we can test whether our forecasting model generalises and whether it can recover relationships we plant on purpose.

**Hard fact:** the source data contains **no sales anywhere**. All three datasets below are simulated. What is real is price, promotion, weather, calendar, store location, and competitor proximity — those are the inputs; quantity is generated.

---

## 1. What we are building (locked)

| Decision | Value |
|---|---|
| Forecast target | **Store-level replenishment** — SKU × store × week |
| Chain | PARKnSHOP only |
| Product | **120 true-dairy SKUs** in **10 types** (UHT 36, Chilled milk 25, Cheese 16, Yoghurt 16, Butter 6, Evaporated 5, Condensed 5, Cheese spread 4, Yoghurt drinks 4, Cream 3) |
| Stores | **140 flagship stores** (Consumer Council directory; carries census-tract + district codes) |
| Time | 2024, 366 days (4 price-snapshot days absent, kept as gaps) |
| Analysis grain | **Weekly** (SKU × store × week ≈ 873k rows per dataset) |
| Demo subset | 40 SKU × 60 store × week ≈ 125k rows |

- **DC-level is out of scope** (decided): the declared target was SKU × week × distribution-centre; we dropped it because we have no store→DC mapping and the constraint story still holds at store level.
- **Universe is the 140-store set**, because the competitor and price-gap files are built on it. The earlier 228-store list is secondary.
- Source files 06/08/09 were deleted by the team (judged useless); 07 (district polygons) was restored. 05 (one cross-chain snapshot) exists but is not used for generation.

---

## 2. The three datasets — and how each is used

All three live on the same grid (140 stores × 120 SKUs × 366 days) and use **independent random seeds and independent generation processes**. Because they are separate random samples, **per-number comparison is meaningless** — we compare only **aggregated metrics** (error distribution, recovery rate, false-positive rate, confidence-interval coverage) across them.

### D1 — Pure noise (negative control)

- **What it is:** every cell drawn independently from one negative-binomial distribution with the *same marginal mean and variance as D2*, plus the same 86.8% "shelf-present" censoring. No time pattern, no store pattern, no seasonality, no promo effect — none of it.
- **How it is built:** `qty ~ NegBin(mean = m, var = θ·m)` for every cell, i.i.d.; `m` and `θ` copied from D2's realised marginals so scale isn't a tell.
- **Intended use:**
  - **False-positive floor / leak gate.** A model's error must never fall *below* σ (the data's own noise level). Below σ is only possible if the model saw something it shouldn't have — that stops the whole harness.
  - **Generalisation baseline.** On pure noise a good model should predict ≈ the mean everywhere; if it "finds" structure here, that structure is spurious.
- **Variables:** mean and variance only. No covariates.

### D2 — Real relationships (primary dataset)

- **What it is:** a realistic demand surface built only from relationships we can defend from the observed data, then drawn with overdispersion and censoring that match real dairy counts. Scaled so annual tonnage lands inside an external sanity band (USDA: ≈26.7 kg dairy per person per year; used only to stop the generator being 10× wrong, never fit to).
- **How it is built (log-linear surface):**
  `log μ = β0 + log(SKU base rate) + log(pack size) + log(store weight) + competition terms + seasonality + weekday + holiday + weather + 2024 trend + observed promo depth`, then `qty ~ NegBin(μ, θ)` with 86.8% censoring.
- **Intended use:**
  - **Primary evaluation.** Does the model beat the seasonal-naive baseline, and does it *recover the known coefficients* (e.g. the planted promo uplift, the seasonality) end-to-end? Recovering a known signal proves the pipeline (features → splits → aggregation) doesn't corrupt it.
  - **Generalisation on realistic data** — the dataset a deployed model would actually face.
- **Key variables we read (kept to a small, interpretable set):** **store size** (`store_weight`, district population), **competitor proximity** (convenience + supermarket counts within 500 m, from teammate file 10), **observed promo depth** (from file 01), and **seasonality/holiday**. *The generator may carry more structure; what we report and test centers on these few.*

### D3 — Planted relationships (detection / retrieval test)

- **What it is:** the full D2 surface **plus 2–3 relationships we add on purpose at deliberately weak effect sizes** (relative μ of +5% / +10% / +20%), so they are partly missed.
- **How it is built:** D2's surface, then planted terms added on top:
  - **Amplified supermarket competition** — extra weight on nearby supermarkets.
  - **Promo × competition interaction** — promo uplift stronger in high-competition stores (a *confounded* relationship: tests whether the model can separate promo from competition).
  - **Spurious-but-included covariate** — a near-null feature, to test **false-positive control** (does the model falsely "find" it?).
- **Intended use:**
  - **Retrieval test.** Can the model recover the planted coefficient (point estimate + confidence interval + coverage)? This is the most informative experiment: the relationship is real-in-the-data but weak/confounded, and the question is *where detection stops*.
  - **Generalisation limit.** Shows the model's behaviour as relationships get harder to see.

### How we compare them

| Question | Answered by |
|---|---|
| Is the harness sound (no leak)? | Noise floor on **all three** (error ≥ σ) |
| Does the model generalise from noise → realistic → planted? | Aggregated error across D1 → D2 → D3 |
| Can it recover what we put in? | D2 (known coefficients) + D3 (planted coefficients), recovery rate + CI coverage |
| Does it control false positives? | D1 (none expected) + D3's spurious covariate |

---

## 3. Generation spec (condensed)

**Shared inputs** (all from cleaned files in `command-v-nexux/`):
- `sku_master.csv` — type, pack size, SKU base rate
- `store_weights_140.csv` — store weight (district population, sums to 1)
- `competitor_features.csv` — convenience / supermarket counts within 500 m (teammate file 10)
- `daily_features_2024.csv` — month, weekday, holiday, rainfall
- `sku_daily_prices_2024.csv` — observed promo flag + depth
- (File 11 multi-chain price gap is an *observed analysis* artifact only, never a generation input.)

**One table, three quantity columns** (`qty_d1`, `qty_d2`, `qty_d3`) on one grid → ~18.4M rows total at daily grain; we **fit weekly** (873k) and use daily only for the recovery experiment on a 30×40 subset. This keeps the full walk-forward (12 refits × 3 datasets × 4 models = 144 fits) to hours, not days.

**D1:** i.i.d. `NegBin(m, θ·m)`, same censoring as D2.
**D2:** log-linear `μ` (§2) → `NegBin(μ, θ)` + censoring, scaled to USDA band.
**D3:** D2 + planted terms at 5/10/20%, same draw + censoring.

Each dataset's config + seed + hash is frozen so a rerun is reproducible.

---

## 4. Analysis protocol

**Real-data findings (no simulation needed — start now):**
- Promo intensity by type (share of SKU-days on offer, mean depth, monthly)
- Price path by type (step changes, count of repriced SKUs)
- Promo calendar shape (festivals / month-end vs flat)
- Brand concentration (38 brands over 120 SKUs)

**Simulated-data checks (labelled as simulated):**
- **Noise-floor gate** — error vs σ on each dataset; below σ = leak, stop.
- **Coefficient recovery** — planted vs recovered, with CI and coverage (primary result).
- **Baseline vs candidate** — seasonal naive, moving average, NB GLM, LightGBM.
- **Constraint layer** — MOQ, pack rounding, shelf life applied to the forecast; delta per type.

**Model choice (and a note on the neural network):** the **primary model is a negative-binomial GLM** (log link, store + SKU fixed effects) because it returns *coefficients* we can compare to the planted truth — that is what the D3 recovery test needs. LightGBM is the challenger (the sponsor's own stack, handles interactions, similar error, but no coefficients). Seasonal naive is the mandatory baseline. A neural network *would* be funny and could fit this data — but it is a black box: it cannot tell us *what* it learned, which defeats the recovery test. So: not recommended, and the GLM is the instrument on purpose.

**The 3–4 variable rule:** analytics holds up best when you can interpret what drives a result. The generator may carry many terms, but what we *report and test* stays at 3–4 interpretable drivers per dataset (store size, competitor proximity, promo depth, seasonality). More than that and the story gets muddy and overfit-prone.

**Pooled, not per-store:** 120 SKUs × 140 stores = 16,800 possible series, most with sparse history. We **estimate one pooled model** (shares the shape — seasonality, promo response — across stores) but **evaluate per store** (each store keeps its own level and its own error metrics). This is the only approach that also forecasts new SKUs with no history.

---

## 5. Status & next step

**Done (cleaning + gating artifacts):** price panel, promo parse (99.99%), SKU master + pack size, calendar, weather, census weights (140), daily context, competitor features, store weights. Generation is **unblocked**.

**Open (one item):** **D3 effect sizes** — which relationships to plant and at what sizes (§2, D3). This is the only thing blocking C3.

**Next:** decide D3's planted terms → write the generator → produce D1 + D2 (weekly) + demo subset.
