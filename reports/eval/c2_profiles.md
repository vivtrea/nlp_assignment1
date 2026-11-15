# C2 · Profile extractor: evaluation

## A) Skill extraction vs. Mehyaar annotated resumes

500 CVs (seed 42). Document-level micro P/R/F1 on lower-cased surface strings. Gold has 102.11 unique skill strings per CV; only **2.0%** of gold annotations are known tech skills (our lexicon). `gold=known` restricts both gold and predictions to known skills (measures recall of real tech skills).

| system | gold | precision | recall | f1 | tp | fp | fn |
|---|---|---|---|---|---|---|---|
| lexicon (ours) | all | 0.948 | 0.014 | 0.028 | 734 | 40 | 50321 |
| lexicon (ours) | known | 0.948 | 0.720 | 0.819 | 734 | 40 | 285 |
| gazetteer from other 4,529 gold CVs (baseline) | all | 0.998 | 0.914 | 0.954 | 46657 | 91 | 4398 |
| gazetteer from other 4,529 gold CVs (baseline) | known | 0.998 | 0.827 | 0.904 | 843 | 2 | 176 |

### Most frequent gold strings the lexicon misses

`com` (394), `skills` (381), `work` (374), `english` (359), `personal` (359), `knowledge` (340), `gmail` (338), `nationality` (287), `hindi` (274), `project` (272), `information` (270), `technical` (268), `professional` (265), `company` (253), `languages` (244), `team` (242), `management` (242), `is` (239), `marital status` (237), `office` (237), `organization` (233), `engineering` (233), `system` (230), `c` (225), `college` (223)

### Most frequent lexicon predictions not in gold

`c++` (28), `c#` (7), `ts` (2), `.net` (2), `headhunting` (1)

## B) Parsing years / English from free text vs. Djinni form fields (1,000 CVs)

| n | years_coverage | years_mae_when_found | years_within_1_when_found | english_coverage | english_acc_when_found |
|---|---|---|---|---|---|
| 1000 | 0.291 | 1.656 | 0.694 | 0.051 | 0.804 |

Used only for resumes without structured fields (fit_labels); Djinni profiles use the form fields directly.