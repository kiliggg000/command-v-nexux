# CLC2026 — Technical Document Plan

The technical document is the reproducibility and audit trail; the deck is the persuasive summary. They must not disagree on a single figure.

**The rules set no page limit on the technical document** — the 20-page cap in §6.2.1 applies to the deck only. The 17.5-page allocation below is a self-imposed target, chosen so the document stays readable rather than because a rule demands it. Depth on §4 and §6 is worth more than brevity here.

---

## 1. Section map

| # | Section | Pages (target) | Content | Evidence we already hold | Status |
|---|---|---:|---|---|---|
| 1 | Problem definition & proposed solution | 2.0 | User, decision, horizon, pain, what the prototype outputs, why it beats current practice | Research brief §1/§4/§5; roadmap Direction A | **Draftable now** |
| 2 | System design | 3.0 | Architecture diagram, data flow in plain language, component table (purpose / input / output / tech / failure check), implemented-vs-mocked labels | Architecture is settled conceptually | **Draftable now** — needs the diagram drawn |
| 3 | Source code & repository access | 1.0 | Public URL, default branch = submitted revision, no-secrets policy, `.env.example`, how to verify access without a request | — | **Blocked** — repo does not exist |
| 4 | Technical implementation | 4.0 | Data processing and quality controls; model and decision logic; baseline; metrics; interpretation | Cleaning is largely done (see `CLC2026_Data_Plan.md` §4.0); model chosen; no results yet | **Partially unblocked** — needs generated data |
| 5 | Setup, install, prototype use | 2.0 | Prerequisites, exact commands, run/test/demo steps, expected output | — | **Blocked** — needs a runnable prototype |
| 6 | Validation plan | 3.0 | Test-case table with pass conditions; split protocol; metric table; failure case; limitations | Framework fully written (E-1…E-5) | **Half ready** — table needs numbers |
| 7 | AI, data, third-party disclosure | 1.5 | Datasets, APIs, models, OSS libraries, licences, AI-tool use, what may not be public | Data list known; AI-use statement needs a decision | **Draftable now** |
| 8 | Team roster & contribution | 1.0 | Names, roles, specific contributions, leader/contact | — | **Needs team input** |

## 2. Section-by-section: what must be in it

### §1 Problem definition & proposed solution
- One named user, one decision, one horizon, one outcome. No solution-first language.
- **The target is store-level replenishment, not DC-level inbound** (decided 2026-10-26). The pain points must be re-derived to match: "scattered demand" and "waste vs stockout asymmetry" survive intact; "early commitment on long import lead time" and supplier-level MOQ do not. See `CLC2026_Data_Plan.md` §3.1.
- A stated claim boundary: what the prototype does not attempt.

### §2 System design
- One diagram: data → validation/cleaning → features → baseline and candidate → constraints → recommendation.
- Component table, every row carrying an **implemented / mocked / proposed** label. With a generated dataset, "mocked" appears in most rows; say it plainly.
- Data flow in ordinary prose as well as the diagram — the guide asks for both.

### §3 Source code & repository access
- Public, opens without an access request.
- Default branch matches the submitted revision exactly.
- `.env.example` present; no keys, tokens, or sponsor-derived values anywhere in history.

### §4 Technical implementation
Four subsections, in this order:
1. **Data processing** — the cleaning rules with reason and row count for each (from the data plan).
2. **Data provenance** — which columns are observed and which are simulated. This is the section that decides whether the document is honest.
3. **Model** — seasonal-naive baseline; **negative-binomial GLM** as primary; **LightGBM** as challenger; constraint layer. No stacking, no LSTM/GRU, no SARIMAX. Justify the GLM on the grounds that a recovery test needs coefficients, not on the grounds that it forecasts better — see `CLC2026_Data_Plan.md` §5. Note the two grains: daily for recovery, weekly for the deliverable.
4. **Interpretation** — what the results do and do not license. Lead with the coefficient-recovery table, not the error table.

### §5 Setup, install, prototype use
- Clean-environment prerequisites, pinned versions.
- Numbered commands that a judge can paste.
- One deterministic demo scenario, plus a fallback if live execution fails.

### §6 Validation plan
- Split protocol: chronological, walk-forward, reserved hold-out, features available at decision time.
- Test-case table: case → input → expected → pass condition.
- Metric table: MAE, RMSE, WAPE, **signed bias reported separately**, each beside the **noise floor** for its variant.
- Leakage gate stated as a test: no error may fall below σ. A baseline comparison is not a leak test — seasonal naive is weak on i.i.d. data and the model will beat it without any leak.
- One mandatory **failure case**, named and explained.
- Limitations, stated as limitations.

### §7 AI, data, third-party disclosure
- Datasets: Consumer Council price archive (2024) · HKO daily weather · HK Government holiday calendar · 2021 Census · PARKnSHOP store locator · **USDA FAS GAIN HK2024-0054** (dairy import tonnage, retail structure) · **OpenStreetMap** for competitor locations. Each with licence and retrieval date.
- **OSM is ODbL** — attribution required and share-alike on derived databases. State it, and keep the derived store-proximity table separate from anything we claim as our own.
- Generated data: declare the generator, its seed, and that every quantity column is simulated. Name the one external anchor it is checked against (retained dairy imports, §1.1 of the data plan).
- OSS: pandas, NumPy, LightGBM/XGBoost, scikit-learn, matplotlib — versions pinned.
- AI tools: state what was used and for what. "None in the pipeline" is only true if the pipeline genuinely calls no model API — the code must match the claim.
- What may not be published, and why.

### §8 Team roster & contribution
- Named contribution per member, specific enough to defend in Q&A. Every member must be able to explain their own part.

## 3. Draftable without data — start here

§1, §2 (minus the diagram render), §7, and §6's *protocol* text. That is roughly 6 pages of the 17.5 and it removes the end-of-week crunch.

Do **not** draft anything carrying a number until the data plan's generation step is frozen. Numbers written early get hand-copied, and hand-copied numbers drift.

## 4. Language rules

| Write | Never write |
|---|---|
| "On the held-out period, MAE was X versus Y for seasonal naive." | "Our model is X% more accurate." |
| "Simulated with stated lead time, MOQ and shelf life, expired stock fell by Z%." | "We reduced waste by Z%." |
| "The dataset is generated; the price and promotion columns are observed." | Any sentence that lets a reader assume the quantities are real |
| "Assumed, not measured: shelf life of N days." | An assumption presented as a finding |

Every metric carries its split, period, aggregation level, and test-set size in the same sentence or an adjacent one.

## 5. Consistency mechanism

One generated `metrics.json` feeds both the deck and this document. No figure is typed twice. If a number cannot be traced to that file, it does not go in.

## 6. Open for discussion

1. **Page budget** — the allocation above is a guess. Does §4 or §6 deserve more at the expense of §2?
2. **Architecture diagram** — hand-drawn SVG in-repo, or a diagramming tool? Affects whether it can be regenerated.
3. **How explicit to be about the synthetic dataset.** Honest and prominent is the defensible choice; the risk is that it reads as "we had no data". Needs a framing decision, not a wording decision.
4. **Does §4 get its own data-quality appendix**, or does that fold into the data plan and get referenced?
