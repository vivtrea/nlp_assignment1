---
name: run-evals
description: Run the evaluation for one or more pipeline components (C1–C6), regenerate reports/eval_summary.md, and interpret the results (high vs low scores, causes, one improvement each). Use after changing any component's code, config or prompts, when asked for metrics, or before saying a task is done.
argument-hint: "[c1 c2 c3 c4 c5 c6 human | empty = all]"
---

# Run evals (CRISP-DM phase 5)

Target components: $ARGUMENTS (empty = all).

## Steps
1. Check that `data/processed/candidates.parquet` exists. If not, run `python scripts/prepare_data.py` first (about 5 minutes).
2. If any C1–C6 code changed, run `python scripts/rescore_labeling_sheet.py` first (re-scores the manually labelled pairs; never edit the `human_*` columns).
3. Run `python eval/run_all.py --only <components>`, or with no `--only` for all. Expect about 10 minutes for everything (C5 is the slowest at about 3.5 minutes). Run it in the background, do not poll it.
4. Read `reports/eval/<component>.md` for each component you ran. Do **not** read the JSON into the conversation unless you need a specific number.
5. For each component, write in `reports/05_evaluation.md`:
   - the headline metric next to its **baseline** (never a system number alone);
   - whether the score is high or low *for this task*, and why (look at the error rows the eval prints);
   - one concrete improvement to try.
6. If a metric moved compared with the previous run, add a dated row to the "Iterations" table in `reports/05_evaluation.md`: change → metric before → after.

## Rules
- Every number must be copied from `reports/eval/*.json` or `*.md`. Never estimate or round differently.
- Never tune on a test set. C1 has a dev set and a held-out set; C5 thresholds and blend weights are tuned on fit_labels **train** only.
- If the LLM path did not run (no `OPENROUTER_API_KEY`), say so explicitly. Do not present rule-based numbers as LLM numbers.
- If a change makes a held-out metric worse, report it; do not hide it.
