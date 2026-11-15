# C1 · Requirement collector: evaluation

LLM enabled: **False**. Without an API key only the rule-based extractor runs.

## A) Slot-filling dialogues (30 hand-written, simulated recruiter)

| extractor | set | completion | turns | extra turns | role acc | years acc | English acc | seniority acc | must-have F1 | nice F1 | skills F1 (any) | LLM fallback rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rules (baseline) | dev (used for tuning) | 1.000 | 3.300 | -0.067 | 1.000 | 1.000 | 1.000 | 1.000 | 0.965 | 1.000 | 0.968 | n/a |
| rules (baseline) | held-out | 0.900 | 3.900 | 0.300 | 0.950 | 0.900 | 0.900 | 0.950 | 0.929 | 1.000 | 0.931 | n/a |

`extra turns` = turns used minus the minimum possible (opening + one answer per question the gold script expects).

### Errors of the rule extractor (16)

| id | slot | gold | pred | set |
|---|---|---|---|---|
| d10 | must | ['Swift'] | ['Swift', 'iOS'] | dev (use |
| d15 | must | ['Laravel', 'MySQL'] | ['Laravel', 'MySQL', 'PHP'] | dev (use |
| d17 | must | ['Kotlin'] | ['Android', 'Kotlin'] | dev (use |
| d24 | must | ['Node.js', 'Vue.js'] | ['JavaScript', 'Node.js', 'Vue.js'] | dev (use |
| d26 | must | ['Pandas', 'Python', 'SQL', 'Tableau'] | ['Pandas', 'Python'] | dev (use |
| h02 | min_years | 1 | None | held-out |
| h02 | english_level | intermediate | None | held-out |
| h04 | must | ['Computer Vision', 'PyTorch', 'Python'] | ['Computer Vision', 'Machine Learning', 'PyTorch', 'Python'] | held-out |
| h09 | role | JavaScript | None | held-out |
| h09 | min_years | 2 | None | held-out |
| h09 | english_level | intermediate | None | held-out |
| h09 | must | ['React Native', 'Redux', 'TypeScript'] | [] | held-out |
| h10 | must | ['Symfony'] | ['PHP', 'Symfony'] | held-out |
| h12 | seniority | unknown | lead | held-out |
| h12 | must | ['CRM', 'Lead Generation'] | ['B2B Sales', 'Lead Generation'] | held-out |
| h19 | must | ['Blockchain', 'Linux', 'Rust'] | ['Blockchain', 'Rust'] | held-out |

## B) Single-shot extraction from 200 Djinni job descriptions vs. form fields

| extractor | role acc | role acc (related) | years exact | years ±1 | years MAE | English acc | English stated in text | English acc when stated |
|---|---|---|---|---|---|---|---|---|
| rules (baseline) | 0.675 | 0.750 | 0.400 | 0.640 | 1.558 | 0.262 | 0.422 | 0.620 |
| majority class | 0.125 |  | 0.400 |  |  | 0.524 |  |  |

Gold = the structured fields the employer filled in on Djinni (independent of the text).
`English stated in text` = share of JDs where an English level could be extracted at all.