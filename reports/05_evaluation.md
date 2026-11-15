# 05 · Evaluation (CRISP-DM phase 5)

Every number is copied from `reports/eval/*.json` (final run of 2026-11-15, **rule-based mode**). Regenerate with `python scripts/rescore_labeling_sheet.py && python eval/run_all.py`. Per-component tables, including error rows, are in `reports/eval/<component>.md`; the one-page view is [eval_summary.md](eval_summary.md).

**Scope note.** The LLM variants (C1 LLM extractor, C2 LLM skills, C5 LLM-as-scorer, C6 LLM writer + LLM judge) are implemented and wired into the evals. A smoke test with a fake LLM executed every LLM code path in the evals without errors. They were **not evaluated** against the real model: the OpenRouter account is on the free tier (50 free-model requests/day), and the LLM evals need about 240 calls. The rule-based versions evaluated here are both the offline system and the baselines the LLM variants will be compared against.

## Why these metrics

| Component | Question the business asks | Metric chosen | Why this metric |
|---|---|---|---|
| C1 | Does the bot collect the right requirements without annoying the recruiter? | slot accuracy, skill P/R/F1, completion rate, extra turns | slots are structured, so exact match and set F1 apply; extra turns measure annoyance |
| C2 | Does the profile contain the skills the CV mentions? | entity P/R/F1 (document-level) | standard for extraction; precision matters most because a false skill becomes a false claim in C6 |
| C3 | Does the filter remove bad fits without losing good ones? | Good-Fit recall, No-Fit removal, precision of kept vs no-filter base rate; labels of the candidates it rejects | a filter is a binary classifier with asymmetric cost: losing a Good Fit is worse |
| C4 | Are relevant candidates near the top? | nDCG@10 (graded), MRR, Recall@50 | graded labels make nDCG appropriate; Recall@50 checks that C5 gets the right pool |
| C5 | Does the final order match fit judgements? | in-pool nDCG@10, Spearman, 3-class macro-F1; **top-3 precision on manual labels** | ranking quality, calibration, and what the recruiter actually sees (the top 3) |
| C6 | Are messages personal and truthful? | faithfulness, verified personal facts, CTA, length, cross-candidate similarity; verifier recall under hallucination injection; manual message score | a hallucinated claim is the main risk, so the verifier itself is tested; faithful ≠ good, so messages are also scored manually |

## Results, interpretation and next step per component

### C1 · Requirement collector

| Set | Completion | Extra turns | Role | Years | English | Must-have F1 | Nice F1 |
|---|---|---|---|---|---|---|---|
| dev (30, used while fixing rules) | 1.000 | −0.067 | 1.000 | 1.000 | 1.000 | 0.965 | 1.000 |
| **held-out (20, never tuned on)** | **0.900** | 0.300 | 0.950 | 0.900 | 0.900 | **0.929** | 1.000 |

One-shot extraction from 200 Djinni JDs vs the employer's form fields: role accuracy **0.675** (0.75 counting related roles) vs majority 0.125; years exact 0.40 (= majority 0.40, MAE 1.56); English accuracy 0.26 vs majority 0.52. English is stated in only **42%** of JDs, and where it is stated, accuracy is 0.62.

- **High:** dialogue slot filling. The held-out set is the honest number: 0.90 completion, 0.93 must-have F1.
- **Low, and why:** (1) role phrases outside the lexicon: "React Native developer" is not mapped, so the bot keeps asking (h09). (2) A bare "upper"/"basic" is not understood when no question is pending. (3) In one-shot JDs the English level is usually not in the text, so the form field can't be recovered: a data limit, not a model limit. Years in text ("3+ years") rarely match the coarse form buckets (1/2/3/5y).
- **Improved in this phase:** taking the role from the job title (job types beat languages) raised one-shot role accuracy from 0.61 to 0.675.
- **Next:** run the LLM extractor (handles unseen phrasings) with rules as a validator; ask a confirmation question when the role isn't detected.

### C2 · Profile extractor

| System | Gold | P | R | F1 |
|---|---|---|---|---|
| lexicon (ours) | all gold labels | 0.948 | 0.014 | 0.028 |
| lexicon (ours) | known tech skills | **0.948** | **0.720** | **0.819** |
| gazetteer from the other 4,529 gold CVs (baseline) | all | 0.998 | 0.914 | 0.954 |
| gazetteer | known tech skills | 0.998 | 0.827 | 0.904 |

- **The gold is the problem, not the extractor.** It has 102 unique "skills" per CV, and only **2.0%** are real tech skills. The most frequent labels are `com`, `skills`, `work`, `personal`, `english`, `gmail`, `nationality`. A gazetteer copied from the other CVs' labels scores 0.95 because it reproduces the annotator's dictionary, not because it finds skills, so "all gold" F1 is not a meaningful target.
- On real tech skills the lexicon is precise (0.95); recall is 0.72, with misses from lexicon gaps, and the gold tagging "c" where the CV says "c++".
- Free-text parsing for resumes without form fields: years found in 29% of CVs (within ±1 year in 69% of those); English found in 5% (accuracy 0.80). This is why the free-text path (fit_labels) gives C5 little evidence.
- **Next:** hand-annotate 50 CVs for an independent gold; measure the LLM skill extractor's precision first.

### C3 · Hard filter

**On fit_labels test** (1,759 pairs, out of domain):

| Must-have coverage | Role rule | Kept | Good-Fit recall | No-Fit removed | Precision of kept |
|---|---|---|---|---|---|
| no filter (baseline) | — | 1.000 | 1.000 | 0.000 | 0.513 |
| 1.0 (strict) | on | 0.169 | 0.114 | 0.837 | 0.530 |
| 0.75 | on | 0.181 | 0.133 | 0.832 | 0.549 |
| 0.5 | on | 0.213 | 0.162 | 0.799 | 0.540 |
| 0.25 | on | 0.274 | 0.245 | 0.748 | 0.552 |

**On manual Djinni labels (in domain):** of the 45 labelled candidates, the current system rejects 8, and **all 8 are labelled No Fit**: the 3 DevOps/SRE engineers for "Junior QA Manual" (removed by the title-based role fix), the 3 QA leads/seniors for the same job (removed by the junior rule), and 2 others that no longer meet the corrected requirements. In the chat demo for a junior manual QA, the junior rule flags 3,764 profiles as over-qualified (many also fail other rules).

- **Low on fit_labels, and why:** precision rises only from 0.513 to ~0.55 while most Good-Fit pairs are lost. These JDs are long, mostly non-Djinni postings: even capped at 6, the extracted must-haves are often skills the resumes describe differently, and the resumes lack form fields. Rejections at strict: must-have 1,390, role 628, years 226, over-qualified 29.
- **In the product setting** (recruiter names 2–4 must-haves; Djinni profiles have form fields) the filter only removed candidates labelled No Fit, and the relaxation ladder guarantees at least 10 candidates (demo case 2: 23 eligible at 75% coverage).
- **Next:** a requirement-confirmation step for pasted JDs; synonym-aware must-have matching.

### C4 · Hybrid retrieval (pooled: 61 queries × 477 resumes)

| Method | nDCG@10 | MRR | Recall@50 |
|---|---|---|---|
| random | 0.031 | 0.117 | 0.146 |
| BM25 (C1 query) | 0.089 | 0.192 | 0.263 |
| LSA dense (C1 query) | 0.086 | 0.197 | **0.315** |
| **Hybrid RRF (C1 query), system** | **0.096** | **0.219** | 0.285 |
| Hybrid RRF (raw JD) | 0.098 | 0.228 | 0.271 |

- Hybrid is best or near-best on the top-of-list metrics: about **3× random** on nDCG@10 and 1.9× on MRR. Dense has the best Recall@50, which supports fusing the two.
- **Why the absolute numbers look low:** unlabelled (JD, resume) pairs count as non-relevant, but most of the 477 resumes were never judged for a given JD. These are lower bounds; the comparison between methods is what matters. LSA explains only 15% of the variance (phase 3).
- **Next:** a neural embedding model (MiniLM) once the environment allows; Recall@k on judged pairs only.

### C5 · Rubric re-ranker

**fit_labels, in-pool** (each test JD's labelled resumes, 61 JDs):

| Method | nDCG@10 | Spearman (per JD) |
|---|---|---|
| random | 0.568 | 0.012 |
| BM25 score | 0.659 | 0.199 |
| hybrid RRF score | 0.682 | 0.232 |
| rubric only (rules) | 0.563 | 0.056 |
| **final: 0.1·rubric + 0.9·hybrid (α tuned on train)** | **0.689** | 0.230 |

3-class labels from the rubric total (thresholds tuned on train): macro-F1 **0.344** vs majority 0.218; global Spearman −0.009.

| Criterion | Weight | Share scored 1 (no evidence) | Spearman vs label |
|---|---|---|---|
| must-have skills | 0.35 | 0.39 | 0.015 |
| relevant experience | 0.25 | 0.78 | −0.014 |
| seniority fit | 0.10 | 0.64 | −0.014 |
| nice-to-have | 0.10 | 0.45 | 0.019 |
| English | 0.10 | 0.98 | −0.008 |
| motivation | 0.10 | 1.00 | n/a |

- **Low on fit_labels, and why:** the rubric reads fields fit_labels resumes don't have (English 98%, motivation 100% and experience 78% get "no evidence → 1"), which flattens the ranking. The blend with retrieval (α swept on train, best 0.1) gives the best test nDCG@10 (0.689) and is the pipeline's final score (approved). Capping one-shot must-haves moved rubric-only nDCG from 0.571 to 0.563, within noise.

**Djinni, manual labels (in domain):** 45 labelled pairs over 10 JDs. The sheet contains every candidate the current system puts in a job's top 3 (all 30 slots labelled), plus earlier picks and weaker candidates.

| Measure | Value |
|---|---|
| **top-3 precision** (system's top 3 labelled Good or Potential) | **0.90** (27 / 30) |
| share of the top 3 labelled Good Fit | 0.47 |
| system #1 labelled Good Fit / Good or Potential | 7 / 10 · 9 / 10 |
| system #1 is the best-labelled candidate of its job | 8 / 10 |
| Spearman, final score vs label, all 45 pairs (rejected candidates count as 0) | 0.451 |
| Spearman, rubric total vs label, all 45 pairs | 0.230 |
| **Like-for-like: the same 30 first-round pairs, old code → new code** | final **0.141 → 0.424**; rubric −0.031 → 0.194 |
| same 30 pairs, only the 25 still retrieved by the new code | final 0.069 |
| labels | Good 18 · Potential 15 · No 12 |

- **What the recruiter sees is mostly right:** 27 of the 30 shortlisted candidates are worth contacting.
- **Remaining misses:** "Head of Payment iGaming": all three are project/programme managers (No Fit). No payments-leadership profile reached the top 3, and the one-shot must-haves (team leadership, statistics, project management, Excel) describe a generic manager. Support Engineer: all three are customer-support rather than technical-support profiles (Potential).
- **Where the gain comes from:** on the same 30 pairs, agreement rises from 0.141 to 0.424, so the improvement is not an artefact of adding pairs. But it comes from the **filter**: the new code rejects 5 of those 30, all labelled No Fit. Among candidates that survive, the order barely tracks the labels (0.069). The system is now good at keeping the wrong people out; sorting the right ones is the open problem, and it's what the LLM scorer is for.
- **Next:** evaluate the LLM scorer on the top 10 (eval code ready, 100 pairs); treat "unknown" fields as missing (renormalise weights) instead of scoring 1.

### C6 · Outreach writer + claim verifier (20 Djinni JDs → 59 messages)

| Writer | Faithful | Claims checked / msg | Verified personal facts / msg | Length in range | CTA | Same-job trigram Jaccard |
|---|---|---|---|---|---|---|
| template (baseline) | **1.000** | 7.83 | 5.34 | 1.000 | 1.000 | 0.459 |

Verifier under hallucination injection (59 messages, one injection of each type):

| Injected claim | Detected |
|---|---|
| fake skill | 1.000 |
| wrong years | 0.881 |
| fake quote | 1.000 |
| fake domain | 1.000 |
| clean message (false-positive rate) | 0.000 |

**Manual message scores (1–5):** the 30 originally labelled messages went from **2.8** (first template) to **3.0** after the template fixes; the messages for the current top-3 candidates average **3.0**.

- **High:** the template never states anything unsupported; the verifier catches every fabricated skill, quote and domain with no false alarms.
- **Fixed in this phase (from the manual review):** restrictions in "Looking For" ("No gambling", "GMT+3 only") are no longer presented as a match; list headers ("…include the following:") and remarks without a goal are no longer quoted; a period inside a number no longer cuts a quote ("…with 4.5 years of experience"); the must-have cap removed job claims like "the role needs Machine Learning" for a full-stack job.
- **Still low, and why:** 3.0/5 is "correct but generic". Same-job messages share many trigrams (0.46) because it's a template; some CV quotes are tool lists. 12% of wrong-year claims pass by design: a profile capped at "10+" accepts any claim ≥ 10.
- **Next:** evaluate the LLM writer (diversity and judge scores, with the verifier enforcing faithfulness); prefer achievement sentences as quotes.

### C7 · UI
Checked with Streamlit `AppTest` (headless): chat → shortlist with evidence → verified messages, in offline and LLM mode; LLM failures degrade to rules with a sidebar warning. The three demo cases were re-run on the final code (see [06_deployment.md](06_deployment.md)).

## Go / no-go
- **Go** for the offline demo and the chat setting: correct slot filling, 90% of shortlisted candidates worth contacting, 100% faithful messages.
- **Caution** for pasted job descriptions from other sources and for roles the candidate pool doesn't contain (payments leadership): one-shot extraction is weaker than the dialogue, and the system can only rank who is in the pool.
- **Next:** run the LLM evals (≥ $10 OpenRouter credit raises the free-model limit to 1,000/day), add a requirement-confirmation step, and grow the labelled set beyond 10 jobs.

## Iterations (loops from evaluation back to earlier phases)

| Date | Trigger (metric) | Went back to | Change | Effect |
|---|---|---|---|---|
| 2026-11-15 | Smoke test: C3 kept 4 of 10,000 | 1 Business / 4 Modeling | relaxation ladder (0.75, 0.5) shown to the recruiter | 23 eligible in the same query |
| 2026-11-15 | C1 dev: completion 0.833, must-have F1 0.840 | 4 Modeling | removed "need" as a must-have cue; clause-level nice-to-have cues; nested skills ("MS SQL" ≠ "SQL") | dev 1.0 / 0.965; then a **held-out set** was written to measure honestly (0.90 / 0.929) |
| 2026-11-15 | C2: lexicon F1 0.03 vs gazetteer 0.95 | 2 Data Understanding | inspected the gold: 2% real skills | report against known skills; gold noise documented |
| 2026-11-15 | C3/C5: 17 of 71 JDs tagged *Support* | 4 Modeling | removed bare generic role words ("support", "sales", "security") | role rejections at strict: 820 → 620 |
| 2026-11-15 | C5: rubric ≈ random on fit_labels | 4 Modeling | rubric + retrieval blend, α tuned on train (approved) | test nDCG@10 0.571 → 0.691 |
| 2026-11-15 | C6 examples: a Lead designer ranked #1 for a Trainee job | 1 Business | over-qualification cap on the experience criterion | unit test |
| 2026-11-15 | Free-tier quota (50/day) vs ~57 LLM calls per search | 4 Modeling | LLM re-scores only the top 10 | ~17 calls per search |
| 2026-11-15 | Manual test: "cv engineer, r&d" not understood | 3 Data Prep (lexicon) | CV/ML-engineer role phrases, R&D domain | query completes with the right role |
| 2026-11-15 | Manual labels: 2 of 10 jobs had only No-Fit candidates; Spearman 0.141 | 4 Modeling (C1) | one-shot role from the job title (job type beats language); must-haves from requirement sections, max 6 | one-shot role accuracy 0.61 → 0.675; "Junior QA Manual" now QA, not DevOps |
| 2026-11-15 | Manual labels: message mean 2.8 / 5 | 4 Modeling (C6) | skip restrictions / list headers / non-goals in "Looking For"; no quote cut inside numbers | 30 original messages 2.8 → 3.0 |
| 2026-11-15 | Manual labels: Junior QA top 3 were leads/seniors (6–11 y) | 1 Business / 4 Modeling (C3) | hard rule (approved): junior jobs exclude senior and lead candidates | new top 3 are 2–4-year QA engineers (2 Good, 1 Potential); top-3 precision 0.90; same-30-pairs Spearman 0.141 → 0.424 (all fixes together) |
