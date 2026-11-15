---
name: crisp-dm-report
description: Update the CRISP-DM phase reports in reports/0X_*.md after any change, and record loops back to earlier phases under "Iterations". Use when finishing a task, when evaluation reveals a data or modelling problem, or when asked to write/update the project report.
argument-hint: "[phase number 1-6 | empty = all touched phases]"
---

# CRISP-DM report keeper

Phases and files (CLAUDE.md §8):
1 Business Understanding → `reports/01_business_understanding.md`
2 Data Understanding → `reports/02_data_understanding.md` (use the `data-understanding` skill for profiling)
3 Data Preparation → `reports/03_data_preparation.md` (numbers from `reports/eval/data_preparation.json`)
4 Modeling → `reports/04_modeling.md`
5 Evaluation → `reports/05_evaluation.md` (use the `run-evals` skill)
6 Deployment → `reports/06_deployment.md`

## Steps
1. Find which phases the change touches (`git diff --stat`, or the task description).
2. Update only the affected sections. Keep existing text; replace outdated numbers with new ones from their source file.
3. When work loops back (for example, evaluation shows a filter is too strict, so you go back to modeling), add a row to the **Iterations** table of the phase you returned to: `date | trigger (phase + metric) | change | effect (metric before → after)`.
4. Keep `docs/architecture.md` in sync if a component's input/output or behaviour changed.
5. Check that every number in the reports comes from a script output. Search the report for numbers you cannot trace, and remove them or link their source.

## Style
- Written for a grader: tables for numbers, short bullets for interpretation, no marketing language.
- State limitations plainly (for example "LLM path not evaluated: no API key").
