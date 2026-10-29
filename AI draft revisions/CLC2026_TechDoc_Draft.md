# Command+v — System Design and Technical Document (Draft)

Team ID T01 · Team Name Command+v · Submission 03/11/2026
Public repository: https://github.com/kiliggg000/command-v-nexux

> **Draft.** Filled for the blank sections. §7 disclosure retained from the existing draft (already
> complete). §8 roster taken from the deck. **Reconcile before submission:** 140-store cohort vs
> "200+ stores"; "50k store-SKU lines"; 30,000 t calibration target vs ~33.7k realised.
> Numbers marked *(proposed)* are placeholders the team should confirm.

---

## Document Checklist

| Required content | Where |
|---|---|
| Problem definition and proposed solution | Section 1 |
| System architecture and technical approach | Sections 2–4 |
| Public repository link and access check | Section 3 |
| Setup, installation, run, test, demo instructions | Section 5 |
| AI, datasets, APIs, open-source components, licences | Section 7 |
| Team roster and contribution declaration | Section 8 |

---

## 1. Problem Definition and Proposed Solution

### 1.1 Problem Context

Dairy was chosen as a sample category with a short shelf life and scattered purchase. Dairy
replenishment relies on store-level demand forecasts, but forecasts often differ from actual sales.
These discrepancies affect ordering decisions and create risks of stockouts, excess inventory, and
waste from expired products.

The intended users are head-office replenishment planners, store managers, and the marketing team.
Every Wednesday, head office consolidates sales data from over 200 stores, enters it into an existing
Excel model, and combines it with the marketing team's promotion plans to determine replenishment
quantities for each store and SKU for the following week. Stores receive replenishment daily. Our
project supports the steps of reviewing forecasts, investigating exceptions, and making replenishment
decisions within this workflow.

Checking forecasts individually across many stores and products increases staff workload. We therefore
propose a review list that highlights persistent over- or under-prediction, limited historical
reference for promotions, and missing data, helping staff prioritise their checks. Forecast evaluation
must also account for stockouts, which can cause observed sales to fall below actual demand.

### 1.2 Proposed Solution

We build **two things**:

1. **A dairy demand forecasting model** — our own component, deliberately **swappable**.
2. **A model-agnostic evaluation harness** that decides whether a candidate model is good enough and
   lets us promote a better one when the data supports it.

The forecast feeds a **review assistant**: differences between forecast and actual sales are converted
into interpretable exception signals, producing an **Excel-compatible review list** (persistent bias,
promo-reference gaps, missing data) plus a **SKU detail** view, with **Adopt / Adjust / Ignore**
feedback written back for calibration.

The design point: the *evaluation harness* is the constant; the *model* is a plug-in. We ship a strong,
interpretable default, and can swap in a challenger whenever the harness shows it wins. This keeps the
system honest and future-proof without re-architecting the product.

### 1.3 Assumptions and Constraints

- **Workflow:** the assistant organises forecasts requiring review by store and SKU; head office uses
  the list in weekly replenishment discussions, with store managers helping investigate local
  conditions. Separate store interfaces/permissions still need confirmation with PARKnSHOP.
- **Data:** we do not have internal sales, historical forecasts, inventory records, or promotion
  plans. The prototype uses Hong Kong public datasets and **simulated sales**. Public chain-level
  online prices are not store transaction prices. Online competitor-promotion coverage/timeliness is
  unconfirmed.
- **Simulated results:** demand relationships and fluctuations are assumed. They can test the model,
  alert rules, and workflow, but cannot establish real demand drivers or PARKnSHOP's performance.
- **Discrepancy interpretation:** actual sales may be depressed by stockouts; the assistant flags
  cases for investigation but cannot determine causes.
- **Model scope:** the forecasting model is our deliverable and is expected to be replaced when a
  better candidate wins on the evaluation harness. The harness — not a single model — is the durable
  asset.
- **Out of scope:** automatic ordering, optimal replenishment quantities, real-time competitor price
  monitoring, production deployment, and any commitment to cost/accuracy improvements.

---

## 2. System Design

### 2.1 Architecture Overview

A weekly, Excel-compatible pipeline in five layers:

- **Data layer** — HK public datasets (Consumer Council price/offers, HKO weather, 1823/GovHK
  holidays, C&SD census, store directory) + **simulated sales** (D1/D2/D3) + the **HQ forecast file**.
- **Processing layer** — align to **store × SKU × week**; clean and validate (nulls, missing weeks);
  build covariates and signals.
- **Model + evaluation layer** — the **forecasting model** (swappable) and the **evaluation harness**
  (walk-forward backtest, metrics, recovery/leak/false-positive tests).
- **Review/output layer** — exception signals → ranked **review list** + **SKU detail**, Excel-compatible.
- **Feedback layer** — planners mark **Adopt / Adjust / Ignore**; the log recalibrates thresholds.

```
[HK public data + simulated sales + HQ forecast]
        │
        ▼
[Ingest & align: store × SKU × week]  ── validate (nulls, gaps)
        │
        ▼
[Forecast model (swappable)] ──► [Evaluation harness: WAPE / bias / calibration / recovery / leak]
        │
        ▼
[Exception signals: deviation | persistence | promo-gap | missing-data | stockout-mask]
        │
        ▼
[Review list + SKU detail]  ──► [Planner: Adopt / Adjust / Ignore] ──► feedback log
```

### 2.2 Data Flow

1. **Ingest (weekly):** HQ forecast, actual sales (simulated), promo/holiday calendar, weather, store
   attributes.
2. **Align & validate:** build the store–SKU–week panel; flag nulls and missing weeks (no silent
   zero-fill).
3. **Fit/forecast:** the model produces a point forecast + prediction interval per store–SKU–week
   (walk-forward; no future leakage).
4. **Score:** the evaluation harness computes WAPE, per-store bias, MAE, calibration, and the
   recovery/leak/false-positive checks.
5. **Signal & rank:** convert forecast-vs-actual into exception signals; rank to a Top-N review list.
6. **Emit:** review list + SKU detail in an Excel-compatible format (CSV/XLSX) plus an HTML detail view.
7. **Feedback:** planner marks Adopt / Adjust / Ignore; the log recalibrates thresholds next cycle.

### 2.3 Key Components

1. **Data ingestion & alignment** — *purpose:* clean store×SKU×week panel; *inputs:* public datasets +
   simulated sales + HQ forecast; *output:* aligned panel; *tech:* Python / pandas.
2. **Forecast model** — *purpose:* produce point + interval forecasts; *inputs:* panel + covariates;
   *output:* forecasts per store–SKU–week; *tech:* Python (NegBin GLM default; LightGBM challenger).
3. **Evaluation harness** — *purpose:* judge any candidate model; *inputs:* forecasts + actuals;
   *output:* WAPE/bias/calibration/recovery/leak metrics + promote/keep decision; *tech:* Python.
4. **Exception rule engine** — *purpose:* interpretable signals + thresholds → ranked exceptions;
   *inputs:* forecast-vs-actual + covariates; *output:* ranked review list with one-sentence reasons;
   *tech:* Python rules.
5. **Review list & SKU detail output** — *purpose:* Excel-compatible review list + detail view;
   *inputs:* ranked exceptions; *output:* XLSX/CSV + HTML; *tech:* Python + html/css/js.
6. **Feedback capture** — *purpose:* Adopt/Adjust/Ignore write-back; *inputs:* planner actions;
   *output:* feedback log; *tech:* lightweight UI or Excel sheet.

---

## 3. Source Code and Repository Access

Public repository: https://github.com/kiliggg000/command-v-nexux

- Confirm the URL is publicly accessible without requesting permission.
- Confirm the default branch contains the submitted prototype version.
- Do not include sponsor-provided data, credentials, API keys, passwords, or confidential materials.
- If configuration is required, provide a safe `.env.example` rather than a real secret.

---

## 4. Technical Implementation

### 4.1 Data Processing

Data sources include population and market-pricing data from HK Government census, Consumer Council
online price archives, HKO weather, public holidays, and store directories (see Section 7). Cleaning:
null entries are marked (not silently filled); missing weeks are flagged; identifiers are read as text
to preserve leading zeros. Feature engineering: promo type/depth, effective unit price, weekend/holiday
flags, weather covariates, competitor proximity counts, and district demographics. Storage: all public
or simulated data is stored in the public repository for third-party checking.

### 4.2 Model, Analytics, or Decision Logic

We forecast **weekly units per store–SKU**. The logic has three layers.

**(a) Forecast model (swappable).** Default: **Negative-Binomial GLM with partial pooling**
(hierarchical), log link:

```
log μ_{i,t} = α + u_i + v_j + β'x_{i,t} + s(t)
u_i ~ N(0, σ_u²)          # store random effect (partial pooling)
v_j                       # SKU effect
x_{i,t}                   # promo, weather, calendar, competition, district census
s(t)                      # stable seasonal term (week-of-year / Fourier) + slow trend
y_{i,t} ~ NegBin(μ_{i,t}, r)
```

- **Partial pooling** is deliberate: 16,800 short series would overfit if fitted separately; shrinkage
  shares shape across stores while keeping per-store offsets. It also lets the model predict **unseen
  stores** (needed for the store-holdout test).
- **Negative-Binomial** handles overdispersion (var = μ + μ²/r) and yields prediction intervals
  directly.
- **Stable seasonality + slow trend:** the seasonal term is regularised; the trend is updated by
  walk-forward retraining, so the model adapts to taste drift without chasing weekly noise.

*Challenger:* LightGBM (accuracy/interactions; no coefficients). *Baseline:* seasonal-naive +
moving-average (must-beat floor).

**(b) Evaluation harness (the constant).** Model-agnostic: every candidate is scored on the same
walk-forward backtest with WAPE, per-store bias, MAE (store-aggregate), interval calibration,
coefficient recovery (D2), leak gate (D1), and false-positive control (D3). A challenger is promoted
only if it beats the incumbent on WAPE **and** passes calibration/recovery. This is what makes the
model swappable.

**(c) Review logic.** Forecast-vs-actual differences become interpretable exception signals
(persistent bias over 8 weeks, promo-with-limited-history, missing data, stockout-masked sales),
ranked into a review list. Outputs: point forecast + interval + probabilities (e.g. P(stockout)) plus
the review list.

### 4.3 Technologies and Dependencies

| Component | Technology / Version | Purpose |
|---|---|---|
| Model + evaluation engine | Python 3.13 (statsmodels / custom GLM; LightGBM) | Fit, forecast, score |
| Data processing | pandas / numpy / pyarrow | Panel build, features, Parquet |
| Excel-compatible output | openpyxl / xlsxwriter | Review list + SKU detail |
| Dashboard | html / css / js | Interaction with tools |
| Storage / Auth | — | Integrate with existing internal tools |

---

## 5. Setup, Installation, and Prototype Use

*(Commands below are proposed; adjust to the final filenames.)*

### 5.1 Prerequisites

Python 3.13; a virtual environment; no paid services. Hardware: a standard laptop (the prototype is
small enough to run locally).

### 5.2 Installation Steps

```bash
git clone https://github.com/kiliggg000/command-v-nexux.git
cd command-v-nexux
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5.3 How to Run, Test, and Demonstrate the Prototype

```bash
python generate_datasets.py                    # regenerate D1/D2/D3 (seed 20261027)
python run_evaluation.py --model negbin_glm    # fit + walk-forward backtest + recovery/leak tests
python run_evaluation.py --model lightgbm      # challenger benchmark
python build_review_list.py                    # emit Excel-compatible review list + SKU detail
```

- **Open:** `dashboard/index.html` (or `python -m http.server` in `dashboard/`).
- **Sample scenario:** load one week's forecast vs actual for a store; inspect the Top-N review list;
  open a SKU detail; record Adopt/Adjust/Ignore.
- **Expected output:** a ranked review list with one-sentence reasons and a SKU detail view.
- **Run tests:** `pytest` (unit tests for alignment, metrics, and rule thresholds).

---

## 6. Validation Plan

Three tests, each with a clear pass condition, run through the same walk-forward harness.

### 6.1 Generalisation (over-fit)

- **Split:** by **time** (hold out the latest month) and by **store** (hold out a stratified sample of
  stores). Fit on the rest; predict the held-out period/stores.
- **Metrics:** WAPE, **per-store bias**, MAE on the store aggregate.
- **Pass:** per-store bias within ±5% (not only aggregate — aggregate bias can mask offsetting store
  errors); WAPE reported and below the seasonal-naive baseline.
- **Caveat:** the store-holdout is only valid if the model pools across stores. Partial pooling gives
  unseen stores a group-level prediction. A store-fixed-effects variant cannot predict unseen stores,
  so for that variant we use the time-holdout only.

### 6.2 Calibration (over-confidence)

- **Split:** same held-out data as 6.1.
- **Steps:** compute the model's 80% and 95% prediction intervals; measure empirical coverage on
  unseen data across a range of inputs.
- **Pass:** empirical coverage ≈ nominal (e.g. 80% ± 5 points). Over-confident = coverage well below
  nominal; under-confident = coverage well above.

### 6.3 Relationship recovery (hallucination)

- **Datasets:** D1 (pure noise), D2 (known relationships), D3 (planted + spurious covariate).
- **D1:** the model must report **no** relationships, and its error must **not** fall below the noise
  floor σ (leak gate).
- **D3:** the model must **recover** the planted effects and must **not** flag the spurious covariate.
- **D2:** recovered coefficients must lie within the confidence interval of the known truth.
- **Pass:** no hallucinated relationships in D1; planted effects recovered in D3; spurious not
  significant.

### 6.4 Model-swap rule

Promote a challenger **iff** it improves WAPE on the incumbent by a margin **and** passes 6.2–6.3.
Otherwise keep the incumbent. Record incumbent, challenger, and swap history.

---

## 7. AI, Data, and Third-Party Disclosure

Retain the existing disclosure table from the current draft (already complete). **One edit:** add
"forecast model + evaluation harness" to the *Purpose* column of the AI-tool and synthetic-data rows,
so the model work is disclosed alongside data processing.

---

## 8. Team Roster and Contribution

| Member | Role |
|---|---|
| CHEONG Kai Lyn Trinity | Data & Rules, Technical, Programming |
| DONG Guoxin | Technical Documentation & Presentation |
| ZHANG Xinwen | Data Collection & Source Documentation |
| LU Yiyang | Business Research & Workflow Analysis |
| GAO Shichen | Solution Planning & Use Cases |

Team Leader: CHEONG Kai Lyn Trinity (kltcheong@connect.ust.hk)
GitHub: https://github.com/kiliggg000/command-v-nexux
