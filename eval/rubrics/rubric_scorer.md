# Rubric: candidate–job fit (human labels for C5 judge agreement)

For each (job, candidate) pair in `data/eval/labeling_sheet.csv`, read the job requirements and the CV, then fill `human_fit`:

- **Good Fit**: meets all must-haves with evidence in the CV, experience and level match; you would contact this person.
- **Potential Fit**: meets most must-haves or is one level off; worth a screening call if the pool is small.
- **No Fit**: misses key must-haves, wrong role, or clearly wrong level.

Fill `human_message_score` (1–5) for rows with a message, using `eval/rubrics/outreach.md` (average of the four dimensions, rounded).
Do not use gender, age or nationality. Label without looking at the system's score column (hide it).
