# CLC2026 — Meeting Prep & Data-Analytics Primer

*For: Nexux. You know K-means, Bayesian decisions, and MLP. This doc explains the
forecasting/evaluation concepts you'll need, and rebuilds your draft agenda so you can
lead the next meeting with understanding — not just a conclusion you don't follow.*

---

## 0. The meeting skill you're building

A weak meeting opens with a *conclusion* ("we should use model X"). A strong meeting opens
with the **menu** — every option that could fit the data, each with its definition, focus,
formula, quirks and problems — and puts the **recommendation last**, for the room to confirm:

- **Understanding** = the why, in plain words, so you can defend it.
- **Option (from the menu)** = each candidate with trade-offs, not one answer.
- **Question** = what only the room can decide.
- **Decision** = one thing you leave the meeting with locked.

For every topic below I've written it in that order. Your job in the meeting: read the
"Understanding," pick which proposal you favour, and ask the "Questions."

---

## PART 1 — Plain-language analytics primer (mapped to what you know)

### 1.1 From "ML you know" to "forecasting we're doing"

| You know | What it is | Our task is the cousin called… |
|---|---|---|
| K-means | Unsupervised grouping | **Segmentation** (optional: group stores by demand shape) |
| Bayesian decisions | Posterior = a *distribution* over outcomes | **Prediction intervals** (we report a range, not one number) |
| MLP | Flexible black-box regression/classifier | **Forecasting** = regression on time, with a hard "no future" rule |

The one big new rule: **forecasting can never use information from the future to predict
the past** (look-ahead bias). That single rule changes how we split data (§1.3) and how
we test the model (§1.5). Everything else is regression you already understand.

### 1.2 The model families that fit our data (the menu, not a verdict)

Our data has a specific shape, and that shape *rules some models in and some out* before we
even compare accuracy:

- **Counts, not continuous** — units sold are non-negative integers → count models (Poisson / NegBin), not plain linear regression.
- **Heavy zeros** — ~87.5% of SKU×store×day cells are zero (still sparse weekly) → zero-aware models help.
- **Hierarchy** — 140 stores × 120 SKUs = 16,800 series, most short → we need *pooled / shared-shape* estimation, not 16,800 separate fits.
- **Covariates** — promo, weather, calendar, competition are daily → the model must eat features.
- **Recovery test** — we must read coefficients back out → the *primary* model must be interpretable (rules out pure black boxes as primary).

That last point is the only hard constraint; everything else is a trade-off. The full menu —
definitions, focus, formulas, quirks, and how each fits *these* formats — is in **Topic 2b**.
Bring that menu to the meeting; decide the winner there, don't pre-announce it.

### 1.3 Train / Validation / Test — and why your ML instinct is *wrong* here

In normal ML you shuffle randomly. **In forecasting, random shuffle leaks the future:**
a Tuesday in March could land in "train" and the same week's Friday in "test," so the model
trains on next-door future data. Instead we split **by time**:

```
2024 weeks:  [======== train ========|== val ==|=== test ===]
               learn patterns          tune      report honest
                                          model     error ONCE
```

- **Train** — fit the model's parameters.
- **Validation** — choose *which* model / settings (GLM vs LightGBM, promo depth binning).
- **Test** — held out, touched **once**, to report the real number we'd brag about.

**"How old?"** Our data is synthetic 2024. For *real* data the question is concept drift:
demand last year may not match this year (new competitor, inflation). Rule of thumb: use the
most recent **12–24 months**; flag anything older as possibly stale. **Proposal:** split 2024
as train = weeks 1–40, val = 41–46, test = 47–53 (or an expanding-window backtest — see §3).

### 1.4 Evaluation metrics — and the zero-inflation trap

We predict a *quantity* (units sold). Error metrics:

| Metric | Plain meaning | Watch-out for us |
|---|---|---|
| **MAE** | average absolute miss (in kg) | intuitive, robust |
| **RMSE** | like MAE but punishes big misses | sensitive to outliers |
| **MAPE** | % error | **EXPLODES** near zero — and 87.5% of our cells are zero! |
| **WAPE** | sum of misses ÷ sum of actuals | **use this instead of MAPE** |
| **Bias** | are we systematically high/low? | catches a broken pipeline |

**Critical:** with 87.5% zeros, MAPE is meaningless (dividing by ~0). Report **WAPE** and
**MAE on the store-aggregate** (a store always sells *something* daily). This is a point you
can make to sound like you get it.

### 1.5 Confidence = prediction intervals + calibration (your Bayes intuition)

You already think in distributions (Bayesian posterior). "Confidence criteria" = do our
**prediction intervals** actually contain reality the right % of the time?
- 80% interval should trap the truth ~80% of the time → **calibration**.
- For the GLM, the interval comes straight from the Negative-Binomial variance
  (var = μ + μ²/r) — no extra machinery.
- A model that's *over-confident* (claims ±2% but misses by ±15%) fails calibration.

**The leak gate / noise floor (our special audit):** on **D1** (pure noise) the *best possible*
error is the data's own noise σ. If any model scores *below* σ, it saw something it shouldn't
— we stop. This is the single most defensible thing in our project; lead with it.

### 1.6 Model audit & improvement (you already built the harness)

"Ways to audit a model" maps onto the three datasets we generated:
- **Residuals** — are leftover errors just noise, or do they still track promo/weather
  (means we missed signal)?
- **Backtest stability** — does error hold across multiple time windows, or did we get lucky?
- **Leak gate** — error ≥ σ on all three datasets (§1.5).
- **Recovery (D2)** — did we rediscover the planted promo coefficient? That *proves* the
  pipeline (features → split → aggregate) doesn't corrupt the signal.
- **False-positive control (D3)** — D3 includes a *spurious* covariate with zero true effect.
  A good model must **not** report it as significant. If it does, our audit caught a phantom.

These are exactly the experiments the data plan already specifies — you can present them as
"here's how we'll prove the model is honest," which is impressive and true.

---

## PART 2 — Your draft, rebuilt as meeting items

### Topic 1 — Grading: is what we build useful / implementable?

**Understanding (the rubric is real).** The official grading is legitimate: the company funds
work it sees value in and would invest a team to build. So "is it useful / implementable" is
not about hitting an accuracy number — it's: *does our deliverable look like the kind of thing
the business would resource a team to build?* That means (1) it solves a concrete operational
problem, (2) it's buildable with data we actually have, (3) it's presentable to non-analysts.
Our natural fit: **replenishment forecasting that cuts waste / stockouts** — a problem every
grocer pays teams to solve.

**Proposals.**
- A. Anchor the whole pitch on one business problem the company would fund (waste / stockout reduction via better store-level forecasts).
- B. Make the *value story* explicit in the deck: current pain → our method → expected operational gain → what a funded team would build next.
- C. Keep the technical audit (recovery / leak gate) as the proof we're not hand-waving, not as the headline.

**Questions to ask.** *Which operational problem does the company most want solved — waste, stockouts, or labour? What would a funded team build first?*

**Decision needed.** The one value narrative we lead with + whether the dashboard is in scope.

### Topic 2a — Evaluation: train/test/validation split

**Understanding.** See §1.3: temporal split, 3-way purpose, recency/"how old."

**Proposals.**
- A. Fixed split: train wk 1–40 / val 41–46 / test 47–53.
- B. Expanding-window backtest (refit 12 times, rolling forward) — more robust, more compute.
- C. For real data: last 12–24 months only; older = stale.

**Questions.** *Is the deadline data 2024-only (synthetic) or will real history be added? How
many refits can we afford to compute?*

**Decision needed.** Split scheme + how-old window.

### Topic 2b — Evaluation: which model + how to state confidence

**Understanding (our data formats — the constraints).** Before picking, name what every
candidate must face: non-negative **counts**; **~87.5% zeros** at SKU×store×day (sparse even
weekly); a **hierarchy** of 140 stores × 120 SKUs (16,800 short series → need pooled
estimation); daily **covariates** (promo, weather, calendar, competition); and a **recovery
test** that requires reading coefficients back out (so the *primary* model must be
interpretable). Confidence = prediction intervals + coverage/calibration (§1.5), not a
single p-value.

**The menu — candidate models that could work with these formats.** (Definitions, focus,
formula, quirks/problems, fit. Recommendation is *last*, not first.)

#### 1. Seasonal-naive baseline
- **Definition:** predict next period = the same period in the previous cycle.
- **Focus:** the must-beat floor; catches whether any "smart" model adds value.
- **Formula:** ŷ_t = y_{t−s}, with s = seasonal period (e.g. 52 weeks).
- **Quirks & problems:** no covariates (can't credit promo/weather); assumes the pattern repeats exactly.
- **Fit:** ✓ mandatory baseline — cheap and honest.

#### 2. Moving-average baseline
- **Definition:** predict the mean of the last k periods.
- **Focus:** a simple level smoother.
- **Formula:** ŷ_t = (1/k)·Σ_{i=1..k} y_{t−i}.
- **Quirks & problems:** smooths seasonality away; no covariates.
- **Fit:** ⚠ secondary baseline only.

#### 3. Poisson GLM (log link)
- **Definition:** count model with log link; assumes mean = variance.
- **Focus:** interpretable coefficients on counts.
- **Formula:** log μ = Xβ;  y ~ Poisson(μ).
- **Quirks & problems:** our data is **overdispersed** (r = 4) → it underestimates variance, so prediction intervals come out too narrow (over-confident).
- **Fit:** ⚠ usable, but the variance assumption is wrong for us.

#### 4. Negative-Binomial GLM (log link, store + SKU fixed effects)
- **Definition:** Poisson plus a dispersion parameter r.
- **Focus:** counts + overdispersion + **interpretable coefficients** (what recovery needs).
- **Formula:** log μ = β₀ + Σ terms;  y ~ NegBin(μ, r);  variance = μ + μ²/r.
- **Quirks & problems:** 120 SKU + 140 store fixed effects = many parameters; sparse series need pooling/regularisation (one pooled model, per-store levels).
- **Fit:** ✓✓ strong primary candidate.

#### 5. Zero-inflated / Hurdle NegBin (ZINB)
- **Definition:** two parts — a logistic "is this a zero?" stage, then a NegBin for the positives.
- **Focus:** explicitly models excess zeros (our 86.8%).
- **Formula:** P(y=0) = π + (1−π)·f(0);  y>0 ~ NegBin(μ, r).
- **Quirks & problems:** statistically the most faithful to our zeros, but **two sets of coefficients** make the recovery test harder to explain; heavier to fit and to tell a story about.
- **Fit:** ✓ contender, but heavier than #4.

#### 6. Croston / TSB (intermittent-demand)
- **Definition:** separates the forecast of *demand size* from the *demand interval*; built for sparse counts.
- **Focus:** intermittent (mostly-zero) series.
- **Formula:** exponential-smoothed mean size and mean interval (no closed form).
- **Quirks & problems:** purpose-built for our sparsity, but **univariate** — can't natively use promo/weather without extensions; no coefficients.
- **Fit:** ⚠ good for the sparse tail, weak on covariates.

#### 7. ARIMA / SARIMA
- **Definition:** autoregressive + moving-average + seasonal terms.
- **Focus:** univariate time-series structure.
- **Formula:** AR/MA lags + seasonal terms (e.g. (p,d,q)(P,D,Q)_s).
- **Quirks & problems:** needs stationarity; **struggles with 87.5% zeros**; fits per-series (impossible at 16,800); no covariates without SARIMAX.
- **Fit:** ✗ poor fit for our setup.

#### 8. Exponential smoothing (ETS)
- **Definition:** level/trend/seasonal components, exponentially weighted.
- **Focus:** univariate seasonal smoothing.
- **Formula:** additive/multiplicative (L,T,S) components.
- **Quirks & problems:** no covariates, no coefficients, no zero handling.
- **Fit:** ⚠ aggregate-level baseline only.

#### 9. LightGBM / XGBoost (gradient-boosted trees)
- **Definition:** ensemble of decision trees learning nonlinear interactions.
- **Focus:** accuracy + interactions (e.g. promo × competition).
- **Formula:** additive trees (no closed form); fit with strict temporal splits.
- **Quirks & problems:** **black box** (no coefficients → can't be the *primary* recovery model); needs careful walk-forward CV to avoid leakage; can overfit.
- **Fit:** ✓ challenger / benchmark.

#### 10. LSTM / MLP (deep learning)
- **Definition:** neural sequence / dense model.
- **Focus:** complex nonlinear patterns.
- **Formula:** recurrent / dense layers with learned weights.
- **Quirks & problems:** black box, needs far more data than our 53 weeks, easy to leak the future, overkill.
- **Fit:** ✗ not recommended (you already flagged this as a joke).

#### 11. Bayesian hierarchical (partial-pooling mixed model)
- **Definition:** shares information across series via priors; each store/SKU level is shrunk toward a global mean.
- **Focus:** many short series **and** interpretable coefficients **and** built-in uncertainty.
- **Formula:** β_i ~ N(β_global, σ);  y_i ~ NegBin(μ_i, r).
- **Quirks & problems:** the best *statistical* answer to "16,800 sparse series," but heavier tooling (Stan/PyMC) and slower to fit.
- **Fit:** ✓✓ advanced option; the "serious" version of #4.

**Recommendation (our proposal — for the majors to confirm, not a verdict):**
- **Primary: Negative-Binomial GLM (#4).** It fits counts + overdispersion, returns the
  coefficients the recovery test needs, and is explainable to judges. (Bayesian hierarchical #11
  is the upgrade path if we want tighter uncertainty on sparse series.)
- **Challenger: LightGBM (#9).** Benchmark it for accuracy/interactions; report it, but it can't
  be primary because it has no coefficients.
- **Mandatory baseline: seasonal-naive (#1)** + moving-average (#2) / ETS (#8) at aggregate.
- **Note on ZINB (#5):** most faithful to zeros, but its two-coefficient structure complicates
  the recovery story — keep as an optional cross-check, not the headline.
- **Out as primary:** LSTM/MLP (#10) and ARIMA (#7) — black box / zero-handling failure.

**Questions to ask.** *Have the decision-analytics majors decided? Do they want a black-box
challenger benchmarked against the GLM? Is the extra complexity of ZINB or Bayesian hierarchical
worth it for the grading?*

**Decision needed.** Lock the primary model (or confirm "majors decide") + the confidence format (80%/95% interval coverage).

### Topic 3 — Deliverable: plan the total output

**Understanding.** The standing CLC rules imply: ≤20-page deck (PDF), tech doc (no limit),
public repo. Extras (dashboard) are optional and time-costly.

**Proposals by sub-item:**
- **i. Dashboard (if in scope) — the decision is scope/access, not engineering.** Per the
  rubric, a dashboard's key calls are: *what items go on it*, *what a store manager is
  authorised to see*, and *what HQ sees*. A **PPT mockup** of those two role-views is a
  perfectly good deliverable — no app build needed. See the content menu below.
  *Decision: mockup vs skip, and which KPIs per role.*
- **ii. Evaluation / accuracy / audit?** This is the **core**. Build the backtest harness +
  metric report + the D1/D2/D3 recovery & leak tests. Our mock data makes this our strongest
  asset. *Decision: must-do.*
- **iii. Math / algorithms?** Write the modeling section of the tech doc (the GLM equation +
  baseline). We already have the generation spec; mirror it for *inference*. *Decision: must-do.*
- **iv. Scale to per-warehouse imports?** Tension: we earlier scoped **DC-level out of scope**
  (no store→DC map). Proposal: v1 = store-level (locked); v2 = warehouse roll-up *only if*
  grading rewards it and we get a mapping. *Decision: v1 only, v2 conditional.*

**Dashboard content menu (proposal — options, not a verdict).** If we mock it up, the two
role-views differ by *authority*:
- **Store-manager view** (authority: *own store only*): per-SKU forecast vs actual this week;
  promo uplift on their lines; stockout / near-expiry alerts; suggested order qty (MOQ + pack
  rounding + shelf-life cap applied); their store's accuracy vs baseline.
- **HQ view** (authority: *all stores, aggregate*): total tonnage forecast by region/type;
  promotion ROI; category trends; an exception list (stores/SKUs below accuracy threshold);
  the model-audit summary (recovery + leak-gate pass).
- **Both views share:** the forecast **with prediction interval**, not just a point — the honest
  part judges notice.

This is a menu to react to, not a decision; the meeting picks the KPIs and whether to include
the HQ exception list.

**What to finish by when?** See Part 3 (realistic, given the Nov 3 deadline).

---

## PART 3 — Timeline (realistic, given your two hard dates)

> **Two hard dates:**
> - **2026-10-31** — your access to me ends. Anything that *needs me* must be done by then.
> - **2026-11-03** — CLC2026 submission. You submit; I can't help after Oct 31.
>
> **Consequence:** build **finished deliverables, not instructions.** Anything I hand you for
> Nov 1–3 must be copy-paste or a click — never "then run this yourself."

| Date | Focus | Owner |
|---|---|---|
| Oct 28–29 | **This meeting** + lock model / grading / scope | you + team |
| Oct 29–30 | Modeling harness: fit GLM + baseline on D2, walk-forward, WAPE/MAE — *needs your sign-off* | me |
| Oct 30–31 | Recovery (D2 coefficients) + D1 noise-floor gate + D3 false-positive | me |
| Oct 31 | Tech-doc modeling + evaluation sections; deck skeleton | me |
| Nov 1–2 | Deck polish + repo finalize — **you alone** (I'm unavailable) | you |
| Nov 3 | Submit | you |

**Must-do (finish by Oct 31, with me):** harness + recovery + leak gate + tech-doc + deck skeleton.
**Nice-to-have (cut if behind):** dashboard **PPT mockup** (cheap now — see Topic 3i), warehouse scaling, extra planted D3 terms.

> **Sign-off note:** running the harness is *modelling code*. Per your own rule (explicit
> approval before any generative/modelling code runs), I won't run it until you approve the
> model choice **and** the exact metrics. This prep doc is planning only.

---

## PART 4 — One-page summary to send *after* the meeting

> Copy this into the team chat within an hour of the meeting:
>
> **Decisions locked:** (model = __ / grading basis = __ / deliverable scope = __)
> **Split scheme:** train __ / val __ / test __
> **Core deliverable:** modeling + evaluation + tech doc + ≤20pp deck + repo
> **Deferred:** dashboard (PPT mockup if scoped) / warehouse scaling (conditional on time/grading)
> **Next owner + deadline:** harness by Oct 30 (Nexux); docs by Nov 1; submit Nov 3.

---

## Appendix — Glossary bridge (your terms → forecasting)

- **K-means cluster** ≈ a *store segment* with similar demand shape (optional EDA).
- **Bayesian posterior** ≈ our **prediction interval** (uncertainty as a range).
- **MLP** ≈ a black-box forecaster; ruled out as the *primary* model because it can't return coefficients (still fine as a benchmark).
- **Overfitting** (you know it) ≈ a model that "finds" patterns in **D1 noise** — caught by the
  leak gate.
- **Cross-validation** (you know it) ≈ here done **walk-forward by time**, never random.
