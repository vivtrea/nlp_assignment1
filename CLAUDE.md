# Recruiter Assistant: NLP Assignment 1

Context for Claude Code. Read this before every task. Keep the file under about 200 lines. Move long material into `docs/` or into a skill.

## 1. What we are building
A recruiter assistant with four stages:
1. **Collect** job requirements from the recruiter through a dialogue. It asks follow-up questions until every required slot is filled.
2. **Search** a candidate database with business logic. This is not vanilla RAG.
3. **Rank** candidates with an explainable rubric that cites evidence from each CV.
4. **Write** a personal LinkedIn outreach message for each top candidate. Every claim in the message must come from that candidate's CV.

Every component must have its own evaluation and report metrics. The assignment brief is in `docs/assignment.pdf`. Graders score four things: system understanding, a working prototype, per-component evaluation, and how the coding agent was managed.

**Hard rule: this must not be vanilla RAG.** "Embed the query, take the top-k, ask an LLM" is not acceptable as the core of the system. Retrieval is one stage inside a pipeline with structured filtering, rubric scoring and grounding checks.

## 2. Pipeline (keep this in sync with `docs/architecture.md`)
| # | Component | Input → Output | Implementation | Eval metric |
|---|---|---|---|---|
| C1 | Requirement collector | chat turns → `JobRequirements` (pydantic) | LLM with JSON-schema output, slot-filling loop | slot P/R/F1, turns-to-complete |
| C2 | Profile extractor | raw CV → `CandidateProfile` (skills, years, seniority, English, domains) | Djinni structured fields + LLM/regex skill extraction | entity P/R/F1 (Mehyaar NER set) |
| C3 | Hard filter | requirements + profiles → eligible pool | deterministic rules: must-have skills, min years, English level, role | filter precision/recall against qrels |
| C4 | Hybrid retrieval | requirements → top-N candidates | BM25 + dense embeddings, fused with RRF | Recall@50, nDCG@10, MRR (fit_labels test as graded qrels) |
| C5 | Rubric re-ranker | JD + top-N → scored list with evidence | weighted rubric (see §5) blended with retrieval score; rules score all top-N, LLM-as-scorer re-scores only the top `LLM_RERANK_TOP` (10), JSON output | nDCG@10 vs C4, macro-F1 / Spearman on fit labels |
| C6 | Outreach writer | profile + JD + rubric evidence → message | LLM with grounded prompt, then claim verifier | faithfulness %, rubric (personalisation, tone, length, CTA) |
| C7 | UI | — | Streamlit chat app | manual demo scenarios |

Each component lives in its own module under `src/recruiter/`. It exposes one pure function or class with typed input and output, and its evaluation script lives in `eval/`.

## 3. Repository layout
```
CLAUDE.md                 # this file
.claude/skills/           # project skills (see §9)
docs/                     # assignment, architecture diagram, CRISP-DM notes
data/raw/                 # downloaded datasets (git-ignored)
data/processed/           # cleaned samples, indices (git-ignored)
data/eval/                # small hand-made gold sets (committed, < 1 MB)
scripts/download_data.py  # reproducible data download
src/recruiter/            # application code, one module per component
eval/                     # one eval script per component + run_all.py
reports/                  # CRISP-DM reports, eval results (markdown + json)
app.py                    # Streamlit entry point
tests/                    # pytest
```

## 4. Data
Get the data with `python scripts/download_data.py`. Never commit anything under `data/raw` or `data/processed`.
| Name | Path | Use | License |
|---|---|---|---|
| Djinni candidate profiles (EN) | `data/raw/djinni_cvs/` | candidate DB, 210,250 rows. Fields: Position, Moreinfo, Looking For, Highlights, Primary Keyword, English Level, Experience Years, CV, id (+ `CV_lang`, `__index_level_0__`: dropped). `CV` is built from Highlights + Moreinfo; `Experience Years` is capped at 11 (= "10+") | MIT |
| Djinni job descriptions (EN) | `data/raw/djinni_jobs/` | 141,897 rows; realistic job inputs, simulated recruiter dialogues. Form fields (Primary Keyword, Exp Years, English Level) are gold for C1 | MIT |
| resume-job-description-fit | `data/raw/fit_labels/` | ranking + scorer eval (No/Potential/Good Fit, 71 test JDs as graded queries); used instead of TalentCLEF. Report metrics on **test**; tune thresholds/weights on **train** only (476/477 test resumes also appear in train) | unstated: eval only |
| Annotated NER PDF Resumes | `data/raw/skills_ner/` | skill-extraction eval. Gold is noisy (~2% of labels are real tech skills): report against all gold and against known tech skills | MIT (HF card) |

Rules:
- Work on a **sample**: 5–20k Djinni profiles, stratified by `Primary Keyword`. Fix the seed at `SEED = 42`. Save the sample to `data/processed/`.
- Do not load the full datasets into the conversation. Inspect them with `.head()`, `.describe()` or the `data-understanding` skill.
- Profiles are already anonymised. Do not try to re-identify people. Never send protected attributes (gender, age, nationality) to the scorer.

## 5. Business logic for ranking
The rubric is adapted from interview evaluation forms (klearskill.com). Weights live in `src/recruiter/config.py` and must not be hard-coded elsewhere.
- Must-have technical skills: 35%
- Relevant experience (years and role match): 25%
- Seniority fit: 10%
- Nice-to-have skills / domain: 10%
- English level: 10%
- Motivation match (the profile's "Looking For" vs the job): 10%

Each criterion is scored 1–5 and must come with an **evidence quote from the CV**. If there is no quote, the criterion scores 1. Failing a must-have in C3 removes the candidate. It is never just a lower score; the only exception is the relaxation ladder (`FILTER_RELAXATION_LADDER`), which lowers must-have coverage when fewer than `RERANK_TOP_K` candidates survive and must be shown to the recruiter. Junior jobs also exclude senior and lead candidates (`FILTER_OVERQUALIFIED_*`, approved 2026-11-15).

Final ranking order = `RERANK_BLEND_ALPHA` × rubric total + (1 − α) × hybrid retrieval score, both min-max normalised per query (α = 0.1, tuned on fit_labels train). The rubric total and its evidence are still shown per candidate. Change α only after re-tuning on train.

## 6. LLM usage
- Use an OpenAI-compatible client pointed at OpenRouter. The base URL and model id go in `.env` (`OPENROUTER_API_KEY`, `LLM_MODEL`). Never commit `.env`; keep `.env.example` up to date.
- All prompts live in `src/recruiter/prompts/*.md`. No long prompt strings inside Python code.
- Structured outputs are always validated with pydantic, with one retry on a validation error.
- Use temperature 0 for scoring and evaluation. Use 0.7 or lower for message writing.
- Cache LLM calls to disk (`data/processed/llm_cache/`), keyed by model, prompt and input hash, so evals are reproducible and cheap.
- Log every call (component, model, tokens, latency) so traces can be shown in the demo video.

## 7. Evaluation rules
- Every component in §2 has `eval/eval_<component>.py`. Each writes `reports/eval/<component>.json` and a markdown summary.
- `python eval/run_all.py` regenerates `reports/eval_summary.md`. After any change to C1–C6, first run `python scripts/rescore_labeling_sheet.py` so the manually labelled pairs (`data/eval/labeling_sheet.csv`) are scored by the current code; labels are never edited by the agent.
- Always report a **baseline** next to the system. Examples: BM25 only vs hybrid vs hybrid + rubric; keyword slot-filling vs LLM.
- LLM-as-judge needs a written rubric in `eval/rubrics/`, and a judge-vs-human agreement check on at least 20 items.
- Never invent or estimate metric numbers. Every number in a report must come from a script output in `reports/eval/`.
- For low scores, write down the likely cause and one concrete improvement to try.

## 8. Process: CRISP-DM
Work in phases. Write each phase's output to `reports/` and keep it up to date:
1. Business Understanding → `reports/01_business_understanding.md` (scope, users, success criteria)
2. Data Understanding → `reports/02_data_understanding.md` (use the `data-understanding` skill)
3. Data Preparation → `reports/03_data_preparation.md` (cleaning, sampling, schema)
4. Modeling → `reports/04_modeling.md` (component choices and alternatives tried)
5. Evaluation → `reports/05_evaluation.md` (metrics, error analysis, go/no-go)
6. Deployment → `reports/06_deployment.md` (how to run, demo scenarios)

When work loops back to an earlier phase (for example, evaluation shows a data problem), note it in that phase's report under "Iterations". The report must show the loops.

## 9. Skills
Project skills are in `.claude/skills/`. Use them instead of improvising:
- `data-understanding`: profile a dataset and update the Data Understanding report.
- `run-evals`: run component evals, regenerate `reports/eval_summary.md`, interpret high and low scores, and log iterations.
- `crisp-dm-report`: update the phase reports and record loops back to earlier phases under "Iterations".

## 10. Code conventions
- Python 3.12 (3.11+ works). Dependencies go in `requirements.txt` with pinned versions.
- Type hints everywhere. Use pydantic models for anything crossing a component boundary.
- Format and lint with `ruff format` and `ruff check`. Test with `pytest -q`. Run all three before saying a task is done.
- Keep functions small and pure where possible. I/O and LLM calls sit at the edges.
- No notebooks in `src/`. Notebooks are allowed in `notebooks/` for exploration, and must be cleared of outputs before commit.
- No secrets, API keys or personal data in code, logs or commits.

## 11. Git workflow
- **Do not create branches, commit or push unless the user explicitly asks.** Work stays in the working tree on `main` until then.
- When the user asks for a commit: small commits with imperative messages, e.g. `Add RRF fusion to retrieval`.
- Do not commit `data/raw`, `data/processed`, `.env` or large binaries (> 5 MB).
- Before any commit or push: tests pass, ruff is clean, and reports are regenerated if the code they describe changed.

## 12. Definition of done (any task)
1. Code works, and tests are added or updated.
2. The eval for the touched component has been re-run, and its numbers are in `reports/eval/`.
3. The relevant CRISP-DM report and `docs/architecture.md` are updated if behaviour changed.
4. Reply with a short summary: what changed, the metric delta, and any open issues.

## 13. When unsure
Ask before you change the rubric weights, add a new dataset, swap the LLM, or change a component's input/output schema. These choices affect the report and the evaluation.
