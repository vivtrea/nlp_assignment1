# 01 · Business Understanding (CRISP-DM phase 1)

## Problem
A recruiter fills an open IT position. Today they write a job post, scan hundreds of profiles by hand, shortlist a handful, and write a personal LinkedIn message to each. The slow parts are **screening** and **personalised outreach**. The risky parts are **missing a strong candidate** and **sending a message that misstates the candidate's background**.

## Users
- **Primary:** an in-house or agency recruiter hiring for tech roles (developers, QA, DevOps, data, design, PM).
- **Secondary:** the hiring manager, who reviews the shortlist and its evidence.

## Scope (what the assistant does)
1. **Collect requirements in a dialogue.** It asks follow-up questions until four required slots are filled: role, must-have skills, minimum years of experience, and English level. Optional: nice-to-have skills, seniority, domain. A full job description can also be pasted and extracted in one shot.
2. **Search a pre-existing candidate database** of 10,000 real, anonymised Djinni profiles with business logic, not plain similarity:
   - hard rules remove ineligible candidates, and each removal gets a stated reason;
   - hybrid retrieval ranks the eligible pool;
   - a weighted interview-evaluation rubric re-ranks the top candidates and cites CV evidence for every score.
3. **Write a personal LinkedIn message** for each top candidate, grounded in their CV. A verifier checks every claim.

## Business logic kept in scope
The rubric is adapted from interview evaluation forms (klearskill.com): rate each competency 1–5 and back it with evidence.

| Criterion | Weight | Why it matters to a recruiter |
|---|---|---|
| Must-have technical skills | 35% | the non-negotiables of the role |
| Relevant experience (years, role match) | 25% | can do the job without ramp-up |
| Seniority fit | 10% | avoids under- or over-qualified hires |
| Nice-to-have skills / domain | 10% | tie-breaker between qualified people |
| English level | 10% | client communication on Djinni-type roles |
| Motivation ("Looking For" vs job) | 10% | likely to answer the message and accept |

Rules:
- A criterion without an evidence quote scores 1.
- Failing a must-have in the hard filter removes the candidate.
- If fewer than 10 candidates survive, the filter relaxes must-have coverage to 75%, then 50%, and **tells the recruiter**.
- For a **junior** job, senior and lead candidates are removed (over-qualified). Within the rubric, extra years stop counting when a candidate is 2+ levels above the job.
- Final order = 0.1 × rubric + 0.9 × retrieval score, normalised per search (α tuned on training data). The rubric provides the explanation and evidence shown to the recruiter.
- Gender, age and nationality never reach the scorer.

## Out of scope
Salary negotiation, ATS integration, actually sending LinkedIn messages, scheduling, and non-English CVs.

## Success criteria (each tied to an evaluation in phase 5)
| Component | Success means | Metric |
|---|---|---|
| C1 collector | collects every required slot correctly with few extra questions | completion rate, slot accuracy, skill F1, extra turns |
| C2 profiles | finds the candidate's real tech skills | P/R/F1 vs annotated resumes |
| C3 filter | removes clear mismatches without losing good candidates | Good-Fit recall, No-Fit removal, precision vs no filter |
| C4 retrieval | relevant candidates near the top | nDCG@10, MRR, Recall@50 vs BM25-only and random |
| C5 re-ranker | ranking and scores agree with human fit judgements | nDCG@10, Spearman, macro-F1 vs fit labels |
| C6 outreach | messages are personal and contain no false claims | faithfulness, personalisation, CTA, length; verifier recall |

## Constraints
- Free or cheap LLM via OpenRouter. The system must also run **without** an API key: every LLM step has a rule-based fallback, which is also its baseline.
- Reproducible: fixed seed 42, scripted data download, evals regenerate every number.
- Must not be vanilla RAG (see `docs/architecture.md`).

## Iterations
| Date | Trigger | Change |
|---|---|---|
| 2026-11-15 | Phase 5: C3 strict filter kept 4 of 10,000 candidates for a realistic 4-must-have query | Added a relaxation ladder (75%, then 50% must-have coverage) shown to the recruiter |
| 2026-11-15 | Phase 5: a Lead designer ranked #1 for a Trainee role | Business rule: experience stops counting when the candidate is ≥ 2 seniority levels above the job |
| 2026-11-15 | Phase 5: rubric alone ≈ random on fit_labels | Final order blends rubric and retrieval (α = 0.1, tuned on train); approved by the project owner |
| 2026-11-15 | Phase 5 manual labels: the top 3 for "Junior QA Manual" were a QA lead and two seniors (6–11 years); the in-rubric penalty was too weak inside the blend | Hard rule (approved): junior jobs exclude senior and lead candidates. The new top 3 are 2–4-year QA engineers, 2 Good + 1 Potential |
| 2026-11-15 | Free-tier quota (50 requests/day) vs ~57 LLM calls per search | LLM re-scores only the top 10 (~17 calls per search) |
