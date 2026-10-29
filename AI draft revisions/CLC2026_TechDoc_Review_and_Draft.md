# CLC2026 — Tech Doc / Deck Review + §2 System Design Draft

> Working review of `CLC2026_reference/Shortlisting_TechDoc_Example.pdf` and
> `Shortlisting_Deck_Example.pdf` (team draft, WIP). Purpose: flag contradictions, poor wording and
> incomplete sections, and give a ready-to-paste draft for Tech Doc **§2 System Design** ("item 2").
> Last updated: 2026-10-29.

---

## 0. What the team's direction actually is (as written)

A **dairy forecast *review* assistant**, not a new forecasting model:

- Takes the **existing HQ Excel forecast** + **actual sales** (simulated) + public context
  (prices/offers, weather, holidays, census, store directory).
- Outputs an **Excel-compatible review list** flagging store–SKU lines that need attention:
  persistent over/under-prediction, limited promo history, missing key data.
- Plus a **SKU detail** view (past 8-week curve, promo markers, stockout bars) and
  **Adopt / Adjust / Ignore** feedback write-back.
- Logic = **interpretable rules** (stated in §2.1, §4.1, and deck §7).
- Users: **HQ replenishment planner** (primary), **store manager** (collaborator),
  **marketing** (info provider).
- Stated success: stockout days ↓, expiry waste ↓, review labour hours ↓, review hit rate ≥60%.

**The big consequence:** the elaborate "which forecasting model (NegBin GLM / LightGBM / …)" menu in
`CLC2026_Meeting_Prep.md` is **not the core** of this direction. The core is a **rules engine that
triages forecast-vs-actual exceptions**. The model menu is only relevant if the team decides to
*also* produce a forecast — which the docs do not currently say. Flag this to the team before
investing more in model selection.

---

## 1. Contradictions (fix first — they change what §2 must say)

1. **Rules vs trained model.** §2.1 / §4.1 / deck §7 say "interpretable rules to generate the review
   list." But §6 Validation says "**Train 3 different instances of the proposed model** … **Retrieve
   relationship information from the model**." You cannot both "use rules" and "train/retrieve a
   model." **Pick one.** Recommended: rules engine; reword §6 to "**apply the review logic** to each
   dataset" and "**check whether the rule flags** the planted effect."

2. **Review the existing forecast vs build our own.** §1 says PARKnSHOP's forecasts differ from
   sales and we *review* the discrepancies. §6 tests *our model's* WAPE/bias as if we forecast.
   Decide: are we (a) triaging the **existing** forecast, or (b) **also** producing a forecast to
   compare? The draft currently implies both.

3. **Scale mismatch.** Tech doc §7: "**140-store** cohort." Business text: "**over 200 stores**."
   Deck §3: "**50k store-SKU lines**." These don't reconcile (140×120 ≈ 16.8k; 200×120 ≈ 24k; 50k
   needs ~250 SKUs). State which is the modelled cohort vs the real chain vs an illustrative figure.

4. **Repo URL typo.** Header (p.1) says `https://github.com/kiliggg000/command-v nexux` (space);
   §3 says `command-v-nexux` (hyphen). Fix the header to the real URL.

5. **"30,000-ton target" vs realized.** §7 calls 30,000 t the assumed target; the generated data
   actually landed ~33.7k t. Say "30,000 t calibration target" explicitly so it doesn't read as a
   contradiction.

---

## 2. Poor wording / broken bits

- §6: "For each of the below, tests are run on: **The**" — sentence cut off.
- §6: "Bias within 5%, **WAPE (??)**" — unresolved placeholder; pick a threshold or say "WAPE
  reported, no hard target."
- Deck §11: "…[Data Collection & Source Documentation]**\]** [LU Yiyang]" — stray bracket.
- "Data Rules & Technical & Programming" (deck §11) — awkward; suggest "Data & Rules, Technical,
  and Programming."
- `PARKnSHOP` / `ParknShop` / `PARKnSHOP` used inconsistently — pick one.
- Deck §3: "ANALYSE **[Forecasting / optimisation / alert rules]**" — pick **alert rules** (matches
  the rules direction).

---

## 3. Incomplete sections (must fill)

| Location | Status |
|---|---|
| Tech doc **§2 System Design** (2.1/2.2/2.3) | **all blank** ← "item 2", draft in §4 below |
| Tech doc **§4.2 Model/Analytics/Decision Logic** | blank — this is the *heart*; say "interpretable rules + signals" |
| Tech doc **§5 Setup/Installation** | all `[command]` placeholders |
| Deck **§4 Data & Assumptions** | all `[placeholder]` |
| Deck **§5 Solution Overview** | all `[placeholder]` |
| Deck **§6 System Architecture** | all `[placeholder]` |
| Deck **§9 Validation / Feasibility / Risks** | all `[placeholder]` |
| Deck **§10 Roadmap** | all `[placeholder]` |
| Deck **§2 "Evidence to include"** | instructions, not content |

---

## 4. Draft — Tech Doc §2 System Design (paste-in)

> Assumes the **rules-based review assistant** direction. If the team switches to a trained model,
> only §2.3 component 3 changes (rule engine → "model + baseline").

### 2.1 Architecture Overview

The prototype is a weekly, Excel-compatible pipeline in five layers:

- **Data layer** — HK public datasets (Consumer Council price/offers, HKO weather, 1823/GovHK
  holidays, C&SD census, store directory) + **simulated sales** (D1/D2/D3) + the **HQ forecast
  file**. All public/simulated; stored in the public repository.
- **Processing layer** — align records to **store × SKU × week**; clean and validate (nulls, missing
  weeks); build deviation and context signals.
- **Logic layer** — **interpretable exception rules** (persistent bias, promo-reference gap, missing
  data, stockout-masked sales) score and rank each store–SKU line.
- **Output layer** — an Excel-compatible **review list** (Store / SKU / Forecast / Interval / Signal /
  one-sentence reason / suggested action) plus a **SKU detail** view (8-week curve, promo markers,
  stockout bars).
- **Feedback layer** — planners mark **Adopt / Adjust / Ignore**; the log feeds threshold
  recalibration in the next cycle.

*(Insert a simple block diagram: Data → Process → Logic → Output → Feedback.)*

```
[HK public data + simulated sales + HQ forecast]
        │
        ▼
[Ingest & align: store × SKU × week]  ── validate (nulls, gaps)
        │
        ▼
[Signals: deviation | persistence | promo-gap | missing-data | stockout-mask]
        │
        ▼
[Interpretable rule engine: score & rank]
        │
        ├──► [Excel review list: Store/SKU/Forecast/Interval/Signal/Reason/Action]
        └──► [SKU detail: 8-week curve, promo markers, stockout bars]
        │
        ▼
[Planner review: Adopt / Adjust / Ignore] ──► feedback log ──► threshold recalibration
```

### 2.2 Data Flow

1. **Ingest (weekly):** HQ forecast, actual sales (simulated), promo/holiday calendar, weather,
   store attributes.
2. **Align & validate:** build the store–SKU–week panel; flag nulls and missing weeks (do not
   silently zero-fill).
3. **Compute signals:** forecast error; bias persistence over the past 8 weeks; promo-with-limited-
   history flag; missing-key-data flag; stockout-suppressed-sales flag.
4. **Apply rules & rank:** score each store–SKU line with interpretable thresholds; sort to a Top-N
   review list.
5. **Emit outputs:** review list + SKU detail in an Excel-compatible format (CSV/XLSX) plus an HTML
   detail view.
6. **Capture feedback:** planner marks Adopt / Adjust / Ignore; the log is written back and used to
   recalibrate thresholds next cycle.

### 2.3 Key Components

1. **Data ingestion & alignment** — *purpose:* build the clean store×SKU×week panel; *inputs:* public
   datasets + simulated sales + HQ forecast; *output:* aligned panel; *tech:* Python / pandas.
2. **Signal builder** — *purpose:* compute deviation, persistence, promo-gap, missing-data and
   stockout-mask signals; *inputs:* panel + calendars; *output:* signal columns; *tech:* Python.
3. **Exception rule engine** — *purpose:* interpretable rules + thresholds → score and rank;
   *inputs:* signals; *output:* ranked exceptions with one-sentence reason; *tech:* Python rules.
4. **Review list & SKU detail output** — *purpose:* Excel-compatible review list + detail view;
   *inputs:* ranked exceptions; *output:* XLSX/CSV + HTML detail; *tech:* Python + html/css/js.
5. **Feedback capture** — *purpose:* Adopt/Adjust/Ignore write-back for calibration; *inputs:*
   planner actions; *output:* feedback log; *tech:* lightweight UI or Excel sheet.

---

## 5. Recommended next actions

1. **Resolve contradiction #1 (rules vs model)** — it determines §2, §4.2, §6 and deck §7. Recommend
   rules engine; then §6 wording changes from "train/retrieve" to "apply/check."
2. **Fill §4.2** with a one-paragraph "signals + rules + thresholds" description (mirror §2.3
   component 3).
3. **Reconcile the numbers** (140 / 200+ / 50k / 30k t) and the repo URL.
4. **Then** decide whether any *forecasting model* is in scope at all — if not, park the model menu
   and keep D1/D2/D3 purely as validation of the rule engine.
