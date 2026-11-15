# 04 · Modeling (CRISP-DM phase 4)

The diagram, contracts and walkthrough are in [docs/architecture.md](../docs/architecture.md). This report records **what was chosen, the alternatives, and why**.

## Component choices

| # | Chosen | Alternatives considered | Why this choice |
|---|---|---|---|
| C1 | Slot-filling dialogue policy (ask the first missing required slot) + two interchangeable extractors: rules (lexicon of 144 skills with aliases, 45 Djinni roles, regex for years and English) and LLM (JSON schema, pydantic validation, one retry). One-shot JD path: role from the job title (job-type roles beat language roles), must-haves from requirement sections, at most 6 | free-form LLM chat; form UI | The policy is deterministic and testable, so the LLM only extracts. Rules are the baseline and the offline fallback. |
| C2 | Djinni form fields used as-is (role, years, English); skills/domains/seniority from text via the lexicon; optional LLM skill extraction that keeps only skills literally present in the CV | spaCy NER trained on the Mehyaar set | Form fields are cleaner than anything extracted from text. The Mehyaar labels turned out too noisy to train on (phase 5). |
| C3 | Deterministic rules: related-role sets, must-have coverage, years ≥ min, English ≥ required. Unknown English or years pass. Relaxation ladder 1.0 → 0.75 → 0.5 | soft penalties only | A recruiter expects hard requirements to be hard, and each rejection reason is explainable. |
| C4 | BM25 (rank_bm25) + LSA dense (TF-IDF 1–2-grams → TruncatedSVD 256 dims), fused with RRF (k = 60), searched **only inside the C3-eligible pool** | sentence-transformers (MiniLM) | MiniLM could not run here (Intel Mac, torch 2.2 vs transformers 5.x), and LSA runs anywhere. The backend is one class, so swapping it is local (`config.DENSE_BACKEND`). |
| C5 | Rubric re-ranker: 6 criteria, 1–5 each, weights in `config.py`, evidence quote required. Two stages: the rules scorer scores all 50 retrieved candidates; in LLM mode the LLM-as-scorer (temperature 0, protected attributes redacted, every quote string-checked against the CV, else score 1) re-scores the top 10. Final order = 0.1 × rubric + 0.9 × retrieval, min-max normalised per search (α tuned on fit_labels train) | cross-encoder fine-tuned on fit_labels | Explainable, matches interview forms, no training data needed. |
| C6 | Template writer (baseline, grounded by construction; skips restrictions, list headers and non-goal remarks in "Looking For") and LLM writer (temperature 0.7). Both go through a rule-based **claim verifier**; an LLM draft with unsupported claims is regenerated once with feedback, then replaced by the template | LLM self-critique only | The verifier is deterministic and auditable, and its recall is measured by injecting hallucinations. |
| C7 | Streamlit chat with a stage trace, per-criterion evidence and a verification badge per message | Gradio | Quick to build, and shows the trace in the demo. |

## LLM setup
- OpenRouter, OpenAI-compatible client. The model is set in `.env`: `LLM_MODEL=baidu/ernie-5.5-flash-preview:free`, `JUDGE_MODEL=nvidia/nemotron-3.5-super-vl:free`, a different model family for judging to reduce self-preference.
- How the model was chosen: the original default (`llama-3.3-70b-instruct:free`) had been withdrawn from the free tier (404), and `gemma-4-31b-it:free` was rate-limited upstream (3 of 4 calls failed after a minute of retries). 7 free models were probed with 3 real C1 prompts each; ERNIE returned valid JSON 3/3 at about 2 s per call, against 11–21 s for the Nemotron models, with equal extraction quality.
- Robustness: 5 retries with backoff on 429/5xx; any remaining API error (404, 401, 402…) raises `LLMUnavailable`, the component falls back to its rule-based version, the UI shows a warning, and evals report the fallback rate.
- Prompts: `src/recruiter/prompts/c1_extract.md`, `c2_skills.md`, `c5_rubric.md`, `c6_outreach.md`.
- Disk cache keyed by (model, temperature, prompt). Every call is logged to `data/processed/logs/llm_calls.jsonl` with component, tokens and latency.

## Alternatives tried during the project
| Date | Component | Tried | Outcome |
|---|---|---|---|
| 2026-11-15 | C1 | "need" as a must-have cue | Broke dialogues: "Need a PHP developer" filled must-haves with PHP, so the bot never asked. Removed (C1 dev completion 0.833 → 1.0). |
| 2026-11-15 | C1 | sentence-level nice-to-have cue | "React, Redux required, Next.js is a plus" put all skills in nice-to-have. Switched to clause-level. |
| 2026-11-15 | C1/C3 | bare generic words as role phrases ("support", "sales", "security") | 17 of 71 fit_labels JDs were tagged *Support* because they contain "support". Replaced with multi-word phrases. |
| 2026-11-15 | C5 | experience score growing with years without limit | A lead ranked first for a trainee job. Capped when the candidate is ≥ 2 levels above the job. |
| 2026-11-15 | C5 | rubric-only vs blend with hybrid score (α tuned on train) | Blend better on fit_labels (test nDCG@10 0.571 → 0.691). Proposed under CLAUDE.md §13, **approved by the project owner and adopted** (`RERANK_BLEND_ALPHA = 0.1`). |
| 2026-11-15 | C5 | LLM-as-scorer on all 50 retrieved candidates | One search used about 57 LLM calls, more than the free tier's 50 requests/day. Changed to two stages: rules on 50, LLM on the top 10 (`LLM_RERANK_TOP`). The LLM refines the shortlist order but can't pull in a candidate that rules ranked below 10th. |
| 2026-11-15 | C4 | dense = sentence-transformers | Environment incompatibility; LSA used instead. |
| 2026-11-15 | C1 | role = longest matching phrase anywhere in a pasted JD | Manual labels: "Junior QA Manual" was parsed as DevOps (the body mentions a DevOps team). Now the job title decides; among matches a job type (QA, DevOps, Data…) beats a language ("AQA Python" → QA Automation). |
| 2026-11-15 | C1 | every skill in a pasted JD is a must-have | Manual labels: 6–12-item must-have lists (incl. "Machine Learning" for a full-stack job) that no candidate matches. Now must-haves come from requirement sections plus title skills, capped at `ONESHOT_MAX_MUST = 6`; the rest become nice-to-have. |
| 2026-11-15 | C6 | first "Looking For" sentence quoted as a match | Manual labels: "No gambling, No part-time" and "GMT+3 only" were presented as a match. Now restrictions, list headers ("…the following:") and remarks without a goal word are skipped; CV quotes must be a sentence of ≥ 4 words, not a header. |
| 2026-11-15 | C5/C6 | "." ends a sentence anywhere | A quote was cut inside "4.5 years". A period now ends a sentence only before whitespace. |

## Iterations
Loops from evaluation back to modeling are listed in the table above and in `reports/05_evaluation.md`.
