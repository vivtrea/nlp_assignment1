# 06 · Deployment (CRISP-DM phase 6)

## How to run
See [README.md](../README.md). In short: `pip install -r requirements.txt` → `python scripts/download_data.py` → `python scripts/prepare_data.py` → `streamlit run app.py`.

| Mode | How | What changes |
|---|---|---|
| Offline (default) | no `.env` | rule extractor, rule rubric, template writer; fully deterministic |
| LLM | `OPENROUTER_API_KEY` + `LLM_MODEL` in `.env` | LLM slot extraction (C1), LLM-as-scorer on the top 10 (C5), LLM writer (C6); the verifier and template fallback stay on. About 17 LLM calls per search. On OpenRouter's free tier (50 requests/day) that's about 2–3 searches a day; failed calls fall back to rules, and the sidebar warns when that happens |

The sidebar shows which mode is active. In LLM mode the "LLM call log" expander shows the last 20 calls (component, model, tokens, latency, cache hit). The demo video is recorded in offline mode (the free quota was used up), where the per-stage pipeline trace is the trace.

## Demo scenarios (C7 manual evaluation)
Re-run on the final code on 2026-11-15 (offline mode; scenario 1 also headless through Streamlit `AppTest`, no exceptions). Scenario 1: 3 follow-up questions, 12 eligible, top 3 a senior Java developer (8 y), a Java developer (6 y) and a team lead. Scenario 2: no follow-ups, relaxed to 75% coverage, 23 eligible. Scenario 3: the trace shows 3,764 profiles flagged over-qualified; the top 3 are 1–2.5-year QA engineers. All 9 messages faithful. Scenarios 4–5 are covered in the recorded demo.

| # | Recruiter input | What to show | Expected behaviour |
|---|---|---|---|
| 1 | "We're hiring a senior Java developer for a fintech product." → "Spring, Kafka and PostgreSQL. Kubernetes is a plus." → "5 years" → "upper-intermediate" | full flow | C1 asks exactly 3 follow-ups; Kubernetes goes to nice-to-have; trace shows 10,000 → 12 eligible → retrieval → top 10 → 3 messages; each message ✅ |
| 2 | "Senior Python developer, must have Django, PostgreSQL and Docker, AWS is a plus, 4+ years, English B2" (one message) | one-shot slot filling + relaxation | no follow-up questions; C3 keeps < 10 at 100% must-have coverage, so the UI warns that it relaxed to 75% |
| 3 | "Junior manual QA" → "test cases, Jira" → "no experience needed" → "pre-intermediate" | junior role, "no" answers | `min_years = 0`; "over-qualified" appears in the rejection reasons; juniors on top |
| 4 | Paste a long Djinni job description in the sidebar | one-shot extraction | role from the job title; at most 6 must-haves from the requirement sections, the rest nice-to-have |
| 5 | Open a candidate's "evidence" expander | explainability | every criterion has a score, a rationale and either a CV quote or `[profile]` field; a criterion without evidence is 1 |

## Monitoring and maintenance (if deployed)
- Re-run `python eval/run_all.py` after any change to prompts, lexicon, config or model. The summary is regenerated, never typed by hand.
- Watch the share of messages that fall back to the template (C6) and the share of C3 runs that need relaxation. Both mean the pool or the requirements drifted.
- The lexicon (`src/recruiter/lexicon.py`) is the main thing to maintain: new technologies must be added with their aliases.

## Limitations
- Candidate data is from 2021–2023 Djinni and anonymised (no names, so messages open with "Hi there,").
- LSA instead of neural embeddings, because of the environment (see phase 4).
- The LLM path is implemented and smoke-tested (fake LLM) and works against ERNIE, but was **not evaluated** with the real model: the free OpenRouter tier allows 50 requests/day and the LLM evals need ~240. Its eval code runs automatically once quota allows.
