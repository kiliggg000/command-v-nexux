# CLC2026 — Optional / Additional Features (Pitch & Tech-Doc Notes)

> **Status:** OPTIONAL extensions, parked here for the **pitch / Word-doc narrative only**.
> **No code** is written or implied. The core deliverable (fit NegBin GLM / Bayesian hierarchical,
> recovery test, leak gate, WAPE) is unchanged. These features are documented so the team can decide
> whether to include them in the write-up, and so the pitch can show "what a funded team would build
> next."
> Last updated: 2026-10-29.

---

## 0. Why this document exists

The meeting-prep menu (`CLC2026_Meeting_Prep.md`, Topic 2b) lists the *core* model. This file
captures the **additional capabilities** we discussed but deliberately kept out of the first build:

1. Store segmentation (K-means) before / alongside modeling.
2. Warehouse (DC) level aggregation + inventory setting from forecasts.
3. Model monitoring, drift detection, and decision-aware evaluation.
4. Dashboard — store-manager vs HQ access, flags, and inputs.
5. Model governance — when to change / retune / discard the model.
6. Evaluation cadence — continuous evaluation and A/B testing.
7. Model-based segmentation — compare global vs per-store distributions instead of arbitrary features.
8. Evaluator → business-metric mapping (speculative).

Each is written as **implementation detail for a pitch or Word doc** — conceptual steps and business
value, not scripts.

---

## 1. Feature — Store segmentation (K-means) as a modeling structure

### What it is
Cluster the 140 stores into *k* groups by a demand or demographic profile, then fit one model per
cluster — a middle tier between one global model and 16,800 per-store models.

### Business rationale
- Gives the business a readable story: "we treat these *store types* differently" (e.g. high-income
  promo-sensitive vs bulk-UHT). Judges and managers relate to segments far better than to 16,800
  coefficients.
- Captures "across-the-street" heterogeneity at a coarse, explainable level.

### Implementation details (conceptual — for the write-up)
1. **Choose clustering features.** Two options we have data for:
   - *Demand shape*: each store's weekly sales profile / promo responsiveness (derived from the
     generated series).
   - *Demographics*: district income + household size — the 18-district census (file 04) is already
     joined to stores, so demographic segments are essentially free.
2. **Standardize** features (z-score) so no single column dominates the distance metric.
3. **Choose k** via elbow / silhouette on the training window — document the chosen k and the reason.
4. **Label** every store with its cluster.
5. **Model**: either (a) fit a NegBin GLM *per cluster* (cluster effects instead of 140 store
   effects), or (b) keep partial pooling (#11) and use the clusters only for EDA / storytelling.

### Trade-offs to state honestly
- **Hard boundaries**: a store sitting between two clusters is forced into one bucket. Partial
  pooling does this *softly* and is usually the better fit. Recommendation: clusters for
  **explanation**, partial pooling for the **fit**.
- Choosing *k* is a new hyperparameter / meeting decision.
- Tiny clusters → back to overfitting; k = 1 → back to the global model.

---

## 2. Feature — Warehouse / DC-level aggregation & inventory setting

### What it needs
A **store → warehouse (DC) assignment map**. We do *not* have one yet (DC-level was scoped out
earlier). Without it, we can only demonstrate the *method* on "all 140 stores as one warehouse."

### Business rationale
Quantify the case for centralization: aggregating store demands lets the warehouse hold
*proportionally less* safety stock per unit than any single store (diversification / √N effect).

### Implementation details (conceptual)
1. **Reconcile** store forecasts up to the DC: bottom-up sum of the stores feeding each warehouse
   (bottom-up is unbiased at DC level if the store forecasts are unbiased).
2. **Aggregate the demand distribution**, not just the point forecast: DC demand ≈ sum of
   (partly independent) store demands. Mean grows ~N, standard deviation only ~√N, so the
   coefficient of variation shrinks → required safety stock as a % of demand falls.
3. **Set safety stock from the service level**: to hit, say, 95% service, stock to the 95th
   percentile of the aggregate demand distribution. Our prediction intervals supply this directly;
   P(stockout) becomes a configurable input, not a guess.
4. **Report** total DC inventory, safety-stock capital, and expected fill rate.

### Trade-offs / dependency
- Gated on the store→DC map; otherwise demo on aggregated 140.
- Aggregation naturally removes zero-inflation at DC level (a DC always moves volume), so WAPE is
  well-behaved there.

---

## 3. Feature — Monitoring, drift detection & decision-aware evaluation

### What it is
Move evaluation from "report WAPE" to **business metrics + an operating procedure** for when to
retrain or swap the model.

### Business metrics to report (not just accuracy)
- **Fill rate / service level**: % of demand met from stock (target typically 95–99% for key lines).
- **Expected waste %**: forecast-driven spoilage — critical for dairy's short shelf life.
- **Total safety-stock capital**: inventory $ tied up.
- **Forecast Value Added (FVA)**: does the model beat the naive baseline in *business* terms?
- (Internal only) WAPE / MAE for model diagnostics.

### Implementation details (conceptual)
1. **Track rolling WAPE** on the most recent weeks of the test / holdout.
2. **Define a drift threshold** (e.g. rolling WAPE breaches X%, or a covariate shifts — new
   competitor, price change). Breach → trigger a **walk-forward re-fit** (already in the core plan)
   or fall back to the LightGBM challenger.
3. **Decision-aware evaluation**: score the model on fill rate / waste / capital, not only on
   point-error — this is what the business actually pays for.
4. **Ownership**: state who runs the trigger post-deployment (an organizational question for the room).

### Trade-offs
- Adds a threshold + monitoring cadence decision.
- Turns criterion #4 ("don't fail to adapt as tastes change") from a model property into an
  operating procedure.

---

## 4. Dashboard — the operating interface (store-manager vs HQ)

### What it is
The dashboard is where the business metrics (§3) and model governance meet. Two role-views split
by **authority**, not by feature set. It is not a toy — it is the **governance surface** for the model.

### Store-manager view — access / flag / input
- **Access (own store only):** per-SKU forecast vs actual this week; the forecast **with interval**;
  promo uplift on their lines; stockout / near-expiry alerts; suggested order qty (MOQ + pack
  rounding + shelf-life cap applied); their store's accuracy vs baseline; local waste.
- **Flag (raise to HQ):** "forecast looks wrong", "local event / festival", "competitor opened
  nearby", "shelf / display change", "supply issue". Flags are logged, become features, and justify
  overrides.
- **Input (human-in-the-loop):** local knowledge (events, display, staffing), order corrections
  **with a reason code**, confirm / dismiss alerts. This log is gold for retraining and for
  explaining why a manager overrode the model.

### HQ view — access / flag / input
- **Access (all stores, aggregate):** total tonnage by region/type; promo ROI; category trends;
  **exception list** (stores/SKUs below accuracy threshold); model-audit summary (recovery + leak
  gate); DC-level stock.
- **Flag:** set accuracy / service thresholds; approve model changes; own the retrain trigger.
- **Input:** promo calendar, pricing, range changes, service-level targets.

### Business metrics — team targets vs open questions

The team's deck already names four success measures. Treat them as **hypotheses to define**, not
settled KPIs:

- **Stockout days ↓**
- **Expiry waste ↓**
- **Review labour hours ↓**
- **Review hit rate ↑** (≥60% of reviewed items are "actual issues")

**Open questions (settle with team / sponsor before claiming value):**
1. **What counts as an "actual issue"?** (forecast error above a threshold? a stockout? a promo gap?)
   — without a definition, "hit rate" is uncomputable.
2. **Baseline for each metric:** what are today's stockout days / waste / review hours? Without a
   baseline, "↓" can't be measured.
3. **Data source per metric:** POS, inventory, waste log, meeting minutes — we have none internally;
   what will the sponsor expose?
4. **Unit & window:** per week / per store / per SKU? rolling or fixed?
5. **Target values:** is ≥60% hit rate the sponsor's number or ours? What is the cost of a false
   alarm (planner time) vs a miss (stockout)?
6. **Trade-off:** optimise hit rate, labour saved, or business outcome (waste/stockout)? These can
   conflict.
7. **Ownership after launch:** who watches these numbers?

> Deliberately **questions, not answers** — the direction is still WIP.

### Why it matters for the pitch
It mirrors how real chains organize (store manager owns OSA/waste; HQ owns aggregate cost &
exceptions) — which validates the two-view design rather than inventing one.

---

## 5. Model governance — when to change, retune, or discard

### Conditions to change (retrain / swap / adjust)
- **Accuracy drift:** rolling WAPE on recent weeks breaches a threshold.
- **Calibration failure:** the 80% interval covers only ~60% (over-confident) → intervals too narrow
  → under-stocked.
- **Bias drift:** systematic over- or under-forecast.
- **Structural break:** new competitor, price-regime change, store renovation, supply disruption →
  history less relevant.
- **Audit failure:** model starts scoring below the D1 noise floor or "finds" the D3 spurious
  covariate → stop.
- **Business failure:** fill rate below target, waste above budget.

### Change vs discard
- **Change** (retrain / swap challenger / adjust pooling): accuracy or calibration degraded but
  structure holds.
- **Discard** (retire): can't beat the baseline after retrain, a structural break invalidates
  history, or the audit fails irrecoverably.

### How to tweak data inputs / fine-tune
- Add covariates (weather, competitor proximity, census, promo depth).
- Change granularity (daily → weekly) or pooling level (global / cluster / partial).
- Feature transforms (promo-depth binning, holiday flags), regularization strength, dispersion r.
- Retrain cadence / walk-forward window length.
- **Constraint:** every tuning step respects the **time split** — no future leakage.

---

## 6. Evaluation cadence — continuous evaluation & A/B tests

- **Cadence:** weekly rolling backtest (data is weekly); monthly deeper audit (recovery + leak gate
  re-run).
- **Continuous evaluation:** automated rolling WAPE + calibration on the latest window; alert on
  breach → feeds the §5 triggers.
- **Shadow mode (safe first step):** run the model in parallel, don't act on it, compare to actual —
  catches problems without risking stock.
- **A/B test (conceptual for us):** split stores into control (current process) vs treatment
  (model-driven orders), compare business lift (fill rate, waste, sales). Needs volume, time, clean
  assignment; often infeasible in a short challenge — so we *describe* it as the real-world
  validation step, not run it.
- **In our synthetic setting:** "continuous evaluation" = re-running the backtest on newer windows;
  we can simulate it but not live-deploy.

---

## 7. Model-based segmentation (compare global vs per-store distributions)

### The idea
Instead of K-means on arbitrary features, **fit the global model and a model per store, then compare
their distributions.** Stores whose distribution diverges most from the global are the atypical ones —
a principled, prediction-relevant segmentation signal.

### Pipeline
1. Fit the global model + 140 store models (store level, not per-SKU — ~141 fits, heavy but
   reasonable; per-SKU × store would be absurd).
2. Measure divergence (e.g. KL / predictive-distribution distance, or coefficient-vector distance).
3. **Map which features drive the divergence** (which coefficients differ most).
4. **K-means on those discovered features** for subsequent reclassification of new / unseen stores.
5. Assign segments; optionally fit per-segment models.

### Why it beats arbitrary-feature K-means
- Divergence is measured in **model space** — it reflects what actually changes predictions, not what
  merely looks different in raw features.
- The feature-attribution step tells you *why* a store is atypical ("promo-sensitive",
  "weekend-heavy"), which is explainable.

### Trade-offs
- Compute: ~141 fits (store level) is fine; per-SKU × store is not.
- Short per-store series → noisy per-store models → noisy divergence; partial pooling gives a
  smoother, built-in version (shrinkage already measures "how far from global").
- It's a **discovery → reclassification loop**: discover features, then K-means for ongoing
  assignment.

---

## 8. From evaluators to business metrics (speculative — finalize after model & eval)

Our evaluation harness can *suggest* business consequences:
- **Interval width / calibration** → safety stock needed → inventory capital.
- **Bias** → systematic over/under-order → waste vs stockout skew.
- **Store-aggregate WAPE** → fill-rate proxy.
- **Recovery** → confidence the promo effect is real → promo-ROI credibility.
- **Leak gate** → trust that reported accuracy isn't illusory.

Example: "if the 80% interval covers only 60%, the business will stock out ~20% more often than
planned." This is **speculation for the pitch**; we will finalize the mapping after the model and
evaluation are locked.

---

## 9. Summary for the pitch

The full "what a funded team would build next" arc:
- **Segment** stores (model-based divergence → K-means) → explainable strategy.
- **Aggregate to DC** → quantify centralization savings.
- **Dashboard** → the governance surface (store vs HQ authority).
- **Govern** → change / retune / discard rules + continuous evaluation.
- **Monitor & decide** → keep the model honest after launch, mapped to business metrics.

None change the core method; all are natural extensions a grocer would resource.
