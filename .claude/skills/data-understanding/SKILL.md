---
name: data-understanding
description: Profile a dataset (schema, nulls, text-length distributions, label/category balance, duplicates, language) and write the findings to reports/02_data_understanding.md. Use for the CRISP-DM Data Understanding phase, when a new dataset is added, or when asked to explore, inspect or profile data.
argument-hint: "[dataset name under data/raw, or a path to a .parquet/.csv/.jsonl file]"
---

# Data Understanding (CRISP-DM phase 2)

Target: $ARGUMENTS. If it's empty, profile every dataset in `data/raw/`, one at a time.

## Steps

1. **Locate the data.** Resolve the target to files under `data/raw/<name>/`. If nothing is there, tell the user to run `python scripts/download_data.py --only <name>` and stop. Do not download data yourself unless the user asks.

2. **Run the profiler.** Do not read whole files into the conversation:
   ```bash
   python .claude/skills/data-understanding/scripts/profile_dataset.py <path> \
       --name <name> --out reports/data_profiles/<name>.md --json reports/data_profiles/<name>.json
   ```
   Useful flags:
   - `--label-cols "Primary Keyword" "English Level"` forces columns to be treated as categorical.
   - `--text-cols CV` forces text columns.
   - `--sample 20000` profiles a random sample of large files (seed 42).

3. **Read the generated profile**, plus 3–5 example rows (`df.sample(5, random_state=42)`, with long texts truncated to 300 characters). Check that the profile matches what the data actually looks like.

4. **Interpret, don't just copy numbers.** For each dataset, write these sections:
   - **Purpose in the system**: which component (C1–C7 in CLAUDE.md) it feeds or evaluates.
   - **Schema**: the columns that matter, and how each one maps to `CandidateProfile`, `JobRequirements` or the eval format.
   - **Quality issues**: nulls, empty or very short texts, duplicates, mixed languages, label imbalance, leakage risks (for example, the same CV in both train and test).
   - **Implications for Data Preparation**: concrete cleaning, filtering or sampling decisions, each with a one-line reason.
   - **Risks and bias**: protected attributes that can be inferred from the text, and skewed category coverage.

5. **Update `reports/02_data_understanding.md`.**
   - Create it if missing, with a short intro and one `## <dataset name>` section per dataset.
   - Replace that dataset's section if it already exists. Never duplicate it.
   - Link the full profile: `[full profile](data_profiles/<name>.md)`.
   - Add the dated decisions to the `## Decisions log` at the bottom.

6. **Sync CLAUDE.md.** If the real schema differs from §4 of CLAUDE.md (different columns, a different license, unexpected size), tell the user and propose the edit. Do not silently rewrite CLAUDE.md.

7. **Report back** in 5 lines or fewer: row count, the top 3 issues, and the preparation decisions you propose.

## Rules
- Every number you write must come from the profiler output or a command you ran. Never estimate.
- Do not print more than 20 rows, or any full CV text, into the conversation.
- Profiles are anonymised. Do not try to identify individuals.
- Keep the report readable for a grader: tables for numbers, short bullets for interpretation.
