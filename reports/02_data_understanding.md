# 02 · Data Understanding (CRISP-DM phase 2)

This report covers the datasets the Recruiter Assistant uses: what is in them, how good they are, and what that means for data preparation. Every number comes from `profile_dataset.py` (in the `data-understanding` skill) or from an inspection command run on the raw files. The full profiles are linked in each section.

| Dataset | Rows | Role in system | Profile |
|---|---|---|---|
| djinni_cvs | 210,250 | candidate database (C2–C6) | [full profile](data_profiles/djinni_cvs.md) |
| djinni_jobs | 141,897 (profiled on a 50k sample) | realistic job inputs (C1, C6 demo/eval) | [full profile](data_profiles/djinni_jobs.md) |
| fit_labels | 6,241 train / 1,759 test | graded relevance for ranking & scoring eval (C3, C4, C5) | [full profile](data_profiles/fit_labels.md) |
| skills_ner | 5,029 annotated CVs (JSON in zip) | skill-extraction eval (C2) | inspected manually, see section |
| talentclef | not used | replaced by fit_labels (approved) | see section |

## djinni_cvs

**Purpose in the system.** This is the candidate database. C2 turns each row into a `CandidateProfile`. C3 filters on the structured fields, C4 indexes the text, C5 scores it and C6 quotes from it.

**Schema → `CandidateProfile`**

| Column | Kind | Nulls | Maps to |
|---|---|---|---|
| `id` | uuid, unique (0 dups) | 0 | `candidate_id` |
| `Position` | short string | 0 | `title` (seniority cues) |
| `Primary Keyword` | categorical, 41 values | 43 | `role` (hard filter) |
| `English Level` | categorical: upper 78,146 · intermediate 64,915 · fluent 43,102 · pre 18,197 · basic 5,269 · no_english 136 | 485 | `english_level` |
| `Experience Years` | numeric 0–11, median 3.0 | 0 | `years_experience` |
| `Highlights` | text, median 225 chars | 104,721 | evidence text |
| `Moreinfo` | text, median 538 chars | 0 | evidence text |
| `Looking For` | text, median 119 chars | 108,195 | `looking_for` (motivation criterion) |
| `CV` | text, median 751 chars, p95 2,340 | 0 | main text for retrieval and extraction |
| `CV_lang` | always `en` | 0 | not used |
| `__index_level_0__` | pandas artefact | 0 | dropped |

**Quality issues**
- `CV` is built from `Highlights` + `Moreinfo`. On a 20k sample, 52.4% match their concatenation exactly; the rest differ in whitespace or order. If both are indexed, the same text gets counted twice.
- 180 CVs are shorter than 50 characters, too little to extract skills or quote evidence.
- `Looking For` is missing for 108,195 rows (51%). Under the rubric rule "no evidence → score 1", about half the candidates get the minimum motivation score.
- `Experience Years` is capped at 11, which is 8.0% of rows and probably means "10+". It also takes half-year values (0.5, 1.5, 2.5).
- 5,052 `Position` titles (2.4%) are in Cyrillic, e.g. "Разработчик java". `CV_lang` says `en` for every row, so this field is not a reliable language signal.
- `Primary Keyword` is very imbalanced: JavaScript has 33,980 rows and Rust 160, a max/min ratio of 790.
- No duplicate rows and no duplicate CV texts.

**Implications for Data Preparation**
- Sample 10,000 profiles stratified by `Primary Keyword`, with at least 60 per class, seed 42. A purely proportional sample would leave rare roles with fewer than 10 candidates and make them unsearchable.
- Drop CVs under 50 characters: there is nothing to ground a score or message on.
- Index `Position + CV` only, not `Highlights`/`Moreinfo` again, so text isn't counted twice.
- Map missing `English Level` to `unknown`. C3 lets `unknown` pass and C5 gives it the minimum English score, so a missing field doesn't silently remove a candidate.
- Treat `Experience Years == 11` as "≥ 10" in the UI and in messages, so a message never claims "11 years".
- Keep Cyrillic titles. The CV body is English and the role comes from `Primary Keyword`.

**Risks and bias**
- Gender can be inferred from text: 377 of 20,000 sampled CVs use he/she/his/her, and 72 contain "years old". C5 and C6 never receive gender, age or nationality. The scorer prompt sees only the fields it needs.
- The role imbalance means rare roles (Rust, Scala, Salesforce) have small pools even after the floor, so a search for them returns fewer strong candidates.

## djinni_jobs

**Purpose in the system.** These are realistic job descriptions. They feed C1 (single-shot extraction from a JD, plus seeds for the hand-written dialogues), the end-to-end demo, and the C6 outreach eval.

**Schema → `JobRequirements`**: `Position` → `title`, `Primary Keyword` (43 values) → `role`, `Exp Years` (no_exp / 1y / 2y / 3y / 5y) → `min_years`, `English Level` → `english_level`, `Long Description` (median 1,631 chars) → must-have and nice-to-have skills, domain.

**Quality issues**
- On the 50k sample, `English Level` is missing for 2,601 rows (5.2%).
- `Exp Years` is coarse (5 buckets) and stops at "5y", so a JD can't say "8+ years" in the form; the text may.
- `Primary Keyword` is skewed: JavaScript has 6,292 rows in the 50k sample.
- `Company Name` is present. It is not personal data but it is real, so it is not shown in demos.

**Implications for Data Preparation**
- Use the structured form fields (`Primary Keyword`, `Exp Years`, `English Level`) as **gold labels** for C1's role, years and English slots. They were filled in by the job poster, independently of the text C1 reads.
- Map `Exp Years` to integers: no_exp → 0, 1y → 1, 2y → 2, 3y → 3, 5y → 5.

**Risks and bias**: the postings are from 2021–2022 (`Published`), so newer tech stacks are under-represented.

## fit_labels

**Purpose in the system.** Graded human labels (No Fit / Potential Fit / Good Fit) for resume–JD pairs. These evaluate C3 (does the filter keep Good Fit resumes?), C4 (ranking a pool) and C5 (rubric scores against fit labels).

**Schema**: `resume_text` (median 5,080 chars in test), `job_description_text` (median 2,401), `label`.

**Quality issues**
- **Leakage:** 476 of the 477 unique test resumes also appear in train. JDs (0) and pairs (0) do not overlap. We don't train a model, so this matters only if thresholds were tuned on train. In that case the tuned thresholds partly reflect the same resumes, which is noted in the evaluation report.
- Test labels: No Fit 857, Good Fit 458, Potential Fit 444 (max/min 1.9). Only 28 of the 71 test JDs have at least one Good Fit.
- Each JD has between 2 and 89 labelled resumes (median 20). This gives 71 natural queries for reranking.
- The license is not stated, so this dataset is used for evaluation only.

**Implications for Data Preparation**
- Use the **test** split for all reported metrics. Use train only to tune C5's 3-class thresholds.
- Ranking gains: Good = 2, Potential = 1, No = 0. Unlabelled resumes in the pooled setting count as 0, which is noted as a lower-bound bias.

## skills_ner (Mehyaar Annotated NER PDF Resumes)

**Purpose**: gold skill spans to evaluate C2's skill extraction (entity P/R/F1).

**Structure**: `ResumesJsonAnnotated.zip` holds 5,029 JSON files, each with `text` and `annotations` as `[start, end, "SKILL: <surface>"]`. The text is extracted from PDFs and reads out of order: section headers get glued to the previous line, e.g. "2019WORK EXPERIENCE".

**Quality issues**: the gold labels are **noisy**. The first file alone tags "Building", "Knowledge", "Technology", "performance" and "Images" as skills. Precision against this gold therefore has a ceiling well below 1 for any skill-specific extractor. The evaluation measures how noisy the gold is instead of assuming it.

**Implication**: evaluate on a seeded sample of 500 CVs, at document level (the set of normalised skill strings), and report results both against all gold labels and against gold labels that are known technical skills.

## talentclef

Not downloaded: `download_data.py` treats it as a manual step. It is also not needed. `fit_labels` test provides 71 JD queries with graded human relevance, which covers the ranking evaluation (nDCG@10, MRR, Recall@k) that TalentCLEF was meant for. Approved and applied to CLAUDE.md §4 on 2026-11-15.

## Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-11-15 | Sample 10k Djinni CVs, stratified by `Primary Keyword`, ≥ 60 per class, seed 42 | keeps rare roles searchable; fits in memory |
| 2026-11-15 | Drop CVs < 50 chars | no evidence to ground scores or messages |
| 2026-11-15 | Index `Position + CV` only | `CV` is built from `Highlights` + `Moreinfo`; avoids counting text twice |
| 2026-11-15 | Missing English → `unknown` (passes C3, minimum score in C5) | a missing field should not silently remove a candidate |
| 2026-11-15 | `Experience Years == 11` read as "10+" | the field is capped |
| 2026-11-15 | Djinni job form fields are gold for C1 role, years and English slots | independent of the text C1 reads |
| 2026-11-15 | `fit_labels` test used for C3/C4/C5 metrics; train only for threshold tuning | test resumes overlap train (476/477) |
| 2026-11-15 | Use `fit_labels` instead of TalentCLEF for ranking eval (approved) | TalentCLEF is a manual download; `fit_labels` already has graded qrels per JD |
| 2026-11-15 | Skill NER eval on a 500-CV seeded sample, gold noise measured | gold has generic words labelled as skills |

## Iterations
| Date | Trigger (later phase) | What was re-examined | Effect |
|---|---|---|---|
| 2026-11-15 | Phase 5: C2 lexicon F1 0.03 vs gazetteer 0.95 on "all gold" | inspected the NER gold: 102 labels per CV, only 2% real tech skills (`com`, `gmail`, `nationality` are the most frequent) | C2 reported against known tech skills; gold noise documented above |
| 2026-11-15 | Phase 5: 17 of 71 fit_labels JDs tagged *Support* | generic words ("support", "sales", "security") occur in most JDs | role phrases made multi-word (phase 4) |
| 2026-11-15 | Manual labels: a pasted Djinni JD "Junior QA Manual" parsed as DevOps | Djinni JDs mention other teams in the body; the title is the reliable field | one-shot extraction takes the role from the title (phase 4) |
