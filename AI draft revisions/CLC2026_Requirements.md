# CLC2026 — Requirements

What we must submit, what we are judged on, what we may not do.

**Authoritative sources — these files govern, not this one:**
`Registration_Confirmed_for_Corporate_Link_Challenge_2026/Corporate_Link_Challenge_2026_Competition_Rules_and_Submission_Guidelines_v1.6.pdf` (rules) ·
`CLC2026_reference/Shortlisting_Deck_Example.pptx` (deck template) ·
`CLC2026_reference/Shortlisting_TechDoc_Example.docx` (tech-doc template).

This file is a working extract: the rules distilled, plus the mapping from rule → our specific gap. Where the two disagree, the PDF wins. **Rules may be amended — re-check official mail before upload.**

---

## 1. Shortlisting deliverables

| Item | Format | Limit | Filename |
|---|---|---|---|
| Presentation deck | PDF | **≤ 20 pages** (§6.2.1) | `CLC2026_SHORTLISTING_[Team ID]_Presentation_Deck.pdf` |
| System Design & Technical Document | PDF | **No stated limit** | `CLC2026_SHORTLISTING_[Team ID]_System_Design_and_Technical_Document.pdf` |
| Source repository | Public, no access request | — | URL in both PDFs |

- **Deadline: 3 November 2026, 23:59 HKT.** Platform timestamp governs; late or emailed replacements are not accepted.
- Upload to the allocated cloud folder. Stage-folder capacity **10 GB**; over that, email `nexux.ust@gmail.com` to request more.
- Three teams are shortlisted. Awards: HKD 3,000 / 2,000 / 1,000.
- **[Team ID] is still a placeholder.** Confirm from the registration pack before any export.

> **Corrected 2026-10-26:** the page limit applies to the **deck only**. §6.2.2 sets six required content items for the technical document and no page cap. The 20-page target in `CLC2026_Tech_Doc_Plan.md` is therefore self-imposed, not a rule.

## 2. Final round (only if shortlisted)

9 Nov 2026 – 8 Jan 2027. Deadline 8 Jan 2027, 23:59 HKT. Showcase and judging 23 Jan 2027, HKUST.

| Item | Format | Limit | Filename |
|---|---|---|---|
| Final deck | PDF | ≤ 20 pages | `CLC2026_FINAL_[Team ID]_Presentation_Deck.pdf` |
| Final tech doc | PDF | No stated limit | `CLC2026_FINAL_[Team ID]_System_Design_and_Technical_Document.pdf` |
| Demo video | MP4 | ≤ 5 minutes | `CLC2026_FINAL_[Team ID]_Demo_Video.mp4` |

## 3. Judging

The rules publish **four shortlisting dimensions and no weights**:

1. Problem understanding and business relevance
2. Data and technical approach
3. System design and implementation quality
4. Feasibility and validation plan

No point rubric exists. The internal weights below are **ours**, for rehearsal only — never present them as official.

| Dimension | Internal weight | What a 5/5 looks like |
|---|---:|---|
| Problem understanding & business relevance | 30% | Named user, workflow, pain; scope focused |
| Data and technical approach | 25% | Data fit and permissions clear; baseline and method fit; leakage avoided; limits explicit |
| System design & implementation quality | 25% | Coherent architecture; end-to-end path works; reproducible setup |
| Feasibility and validation plan | 20% | Test cases, pass conditions, risks, evidence concrete |

Dimensions 2 and 4 together are 45% of our internal weighting — the two plans in this folder target them directly.

## 4. Hard constraints

| Constraint | Consequence for us |
|---|---|
| Sponsor material is for the registered team only | No sponsor data, field names, or structure to anyone outside the team |
| No sponsor/confidential material, credentials, keys, or passwords in the public repo | `.env.example` only; audit before every push |
| No sponsor material to an external AI service unless LOGFLOWS permits in writing | Governs which tools touch which files |
| Disclose material AI-tool use, third-party datasets, APIs, models, libraries, and licences | Tech doc §7; keep a running list from day one, not at the end |
| Do not falsify results or misrepresent contributions | Includes not presenting simulated output as observed |
| State which parts are **implemented, mocked, or proposed** | With a generated dataset this is mandatory, not optional |

**The one that will bite us:** we have no sales data. Every quantity figure is simulated. The tech doc and deck must say so in the same sentence as the number, or the disclosure rule and the "do not overclaim" instruction are both breached.

## 5. Deck — 12 slides

| # | Slide | Content source |
|---|---|---|
| 1 | Title | Team name, ID, date, scope |
| 2 | Executive summary | Problem, one-sentence solution, expected value |
| 3 | Problem & opportunity | Workflow, pain, stakeholder, success measure — `_archive/CLC2026_Fresh_Food_Demand_Research_Brief.md` §1, §4 |
| 4 | Target user & workflow | INPUT → ANALYSE → ACTION; operator view |
| 5 | Data & assumptions | Fields, period, quality, prep, gaps — `CLC2026_Data_Plan.md` §1–3 |
| 6 | Solution overview | Features, why better than current workflow, one prototype visual |
| 7 | System architecture | Essential components + data flow only |
| 8 | Technical approach | Method, baseline, metrics, validation, **implementation status** |
| 9 | Prototype demonstration | One complete input → process → output path |
| 10 | Validation, feasibility, risks | Test table, failure case, mitigations |
| 11 | Roadmap | Current capability, next 4–8 weeks, longer term |
| 12 | Team & closing | Roles, leader/contact, repo and demo links |

Remove every template prompt and placeholder. Every visual must be readable at presentation size.

## 6. Technical document — 8 sections

**The rules mandate six content items** (§6.2.2.1–6.2.2.6): problem/solution/architecture/design/approach/assumptions · public repo link · technical implementation detail · setup-install-run-test-demo instructions · AI-dataset-API-library-licence disclosure · roster and contribution declaration.

Our 8-section structure is the template's, and covers all six. Depth, evidence, and status per section live in `CLC2026_Tech_Doc_Plan.md`.

1. Problem definition & proposed solution
2. System design (architecture diagram, data flow, components)
3. Source code & repository access
4. Technical implementation (data processing, model, tech stack)
5. Setup, install, prototype use
6. Validation plan (test-case table)
7. AI, data, third-party disclosure
8. Team roster & contribution declaration

## 7. Unconfirmed — treat as absent until written

No written sponsor answers are on record. Design so that absence is survivable.

| Ref | Question | If unanswered, we assume |
|---|---|---|
| D18 | May LOGFLOWS host a certified open-source model in its environment? | No. Phase 1 stays classical, in-environment |
| D3 | Remaining shelf life / receipt date at SKU level? | Absent → shelf-life constraint is **assumed**, and labelled assumed |
| D4 | Lead time, MOQ, pack size? | Absent → constraint layer runs on stated defaults, swept |
| D1 | Data granularity and history depth? | Declared target stays SKU × week × DC |
| G1 | Current manual forecasting process, and how it is judged? | Baseline = seasonal naive, per LOGFLOWS' published incumbent |
| Q6 | Diagnostic or driving mode; actionable precision? | Diagnostic. Report calibration, not just MAE |
| Q7 | Does the rubric tilt workflow or evaluation? | Four dimensions evenly, and say so in the deck |

## 8. Submission gates

- [ ] No unresolved eligibility, file, deadline, access, confidentiality, or disclosure issue
- [ ] Every number has a traceable method and split
- [ ] A judge can understand problem, output, and value without a technical deep dive
- [ ] Prototype completes its core workflow from a clean start, with a backup demo
- [ ] Deck and tech doc agree on every figure, date, and metric name
- [ ] Repo public, opens without a request, matches the submitted revision
- [ ] Both PDFs opened and checked after export; placeholders removed
- [ ] Uploaded early, verified complete, local copy of the exact submitted revision saved

## 9. Open for discussion

1. **Team ID** — still a placeholder. Needed before any export.
2. **Sponsor answers** — did the 13–15 Oct consultation happen? If answers exist anywhere, §7 changes materially.
3. **Deck has no file of its own** in the three-file scheme. §5 keeps it here; confirm that is where it belongs, or the deck becomes a fourth file.
