# 03 · Data Preparation (CRISP-DM phase 3)

Script: `python scripts/prepare_data.py` (seed 42). Every number below is from `reports/eval/data_preparation.json`, which the script writes.

## Candidate database (Djinni CVs → `data/processed/candidates.parquet`)

| Step | Rows | Reason (decision from phase 2) |
|---|---|---|
| Raw | 210,250 | — |
| Drop rows without `Primary Keyword` | 210,207 | role is needed for the C3 role rule and for stratification |
| Drop CVs < 50 chars | 210,008 | no text to ground scores or messages on |
| Drop duplicate CV texts | 210,008 | none found |
| Stratified sample by `Primary Keyword`, ≥ 60 per class | **10,000** | keeps rare roles searchable (Rust, Scrum Master, Flutter, Data Engineer, Lead Generation each get 60); largest: JavaScript 1,309, QA 1,194, Design 717 |

Cleaning applied per row (`c2_profiles.profile_from_djinni`):
- `\r\n` → `\n`; `CV_lang` and `__index_level_0__` dropped.
- `English Level` null → `unknown`. **25** profiles in the sample.
- Junk `Looking For` values ("1", "-", fewer than 5 characters or no word of 3+ letters) → empty. Empty in **5,287** of 10,000 profiles; under the rubric rule these get motivation = 1.
- `Experience Years` kept as a number; 11 is shown as "10+" in the UI and in messages.

## Profile schema (`CandidateProfile`, `src/recruiter/schemas.py`)

| Field | Source | Notes |
|---|---|---|
| `candidate_id` | `id` | uuid |
| `title` | `Position` | used for seniority cues |
| `role` | `Primary Keyword` | C3 role rule |
| `skills` | lexicon on `Position + CV` | median **4** skills per profile; **986** profiles have 0 known skills. Top roles among them: Other 99, Marketing 96, QA 94, Project Manager 89, Support 86, JavaScript 60, HR 58. These are largely non-developer roles whose CVs describe duties rather than tools; such candidates can only pass C3 when the job has no must-have skills. |
| `years_experience` | `Experience Years` | |
| `seniority` | title keyword, else years (< 2 junior, < 5 middle, < 8 senior, else lead) | junior 3,245 · middle 2,991 · senior 2,197 · lead 1,567 |
| `english_level` | `English Level` | |
| `domains` | domain lexicon on text + Looking For | |
| `looking_for` | `Looking For` (cleaned) | |
| `cv_text` | `CV` | evidence source for C5 and C6 |

## Search index (`data/processed/index/hybrid.pkl`)
- **BM25** over `Position + CV`, tokenised with a regex that keeps `c++`, `c#` and `.net`, with stop words removed.
- **LSA dense vectors:** TF-IDF on 1–2-grams (min_df 2, max_df 0.5, sublinear tf) gives a vocabulary of **119,777** terms, reduced by TruncatedSVD to **256** dimensions. These explain **15%** of the variance, which is low and expected for short, diverse CVs. It is one reason dense alone is not enough (see phase 5).

## Evaluation data
| Set | Built by | Size | Use |
|---|---|---|---|
| `data/eval/c1_dialogues.jsonl` | hand-written | 30 dialogues | C1 dev set (used while fixing rules) |
| `data/eval/c1_dialogues_heldout.jsonl` | hand-written after tuning stopped | 20 dialogues | C1 held-out |
| Djinni jobs sample | `eval_c1` (seed 42) | 200 JDs | C1 single-shot vs form fields |
| fit_labels test / train | `eval/fit_data.py` | 1,759 / 6,241 pairs | C3/C4/C5 metrics on test; thresholds and α tuned on train |
| Mehyaar NER | `eval_c2` (seed 42) | 500 CVs | C2 skills |
| `data/eval/labeling_sheet.csv` | `scripts/make_labeling_sheet.py` (seed 7), extended by `scripts/rescore_labeling_sheet.py` | 45 labelled pairs, 10 Djinni JDs | manual labels: C3/C5 in-domain check (top-3 precision), C6 message quality |

## Iterations
| Date | Trigger | Change | Effect |
|---|---|---|---|
| 2026-11-15 | Smoke test: a message quoted `Looking For` = "1" | junk `Looking For` values emptied | empty `Looking For`: 5,185 → 5,287 |
| 2026-11-15 | C1 eval: "MS SQL" also produced "SQL" | lexicon drops a match nested inside a longer skill's span; profiles rebuilt | (re-run of C1/C2 evals) |
| 2026-11-15 | Manual test: "cv engineer (middle), for r&d project" not understood | lexicon: CV/ML/research-engineer role phrases, "cv engineer" alias for Computer Vision, new *r&d* domain (16 domains); profiles rebuilt | query completes with role Data Science, domain r&d |
| 2026-11-15 | Manual labels | labelled pairs re-scored after every code change by `scripts/rescore_labeling_sheet.py`; new top-3 candidates appended for labelling | sheet grew from 30 to 45 labelled pairs (all 30 current top-3 slots labelled) |
