# C3 · Hard filter: evaluation (fit_labels test)

1759 labelled pairs, 71 JDs. Requirements extracted by C1 (rules), profiles by C2 (free text).
Baseline (no filter): keeps 100%, precision of kept = **0.5128** (share of Good/Potential pairs).

Extracted must-have counts per JD: {0: 15, 1: 18, 2: 5, 3: 8, 4: 6, 5: 3, 6: 16}; JDs with a min-years rule: 54; with an English rule: 1.

| must_have_coverage | role_rule | kept_share | good_recall | good_or_potential_recall | no_fit_removed | precision_of_kept |
|---|---|---|---|---|---|---|
| 1.000 | False | 0.193 | 0.127 | 0.194 | 0.809 | 0.516 |
| 0.750 | False | 0.218 | 0.157 | 0.224 | 0.788 | 0.526 |
| 0.500 | False | 0.305 | 0.269 | 0.319 | 0.711 | 0.537 |
| 0.250 | False | 0.431 | 0.437 | 0.463 | 0.602 | 0.551 |
| 0.000 | False | 0.869 | 0.878 | 0.891 | 0.155 | 0.526 |
| 1.000 | True | 0.169 | 0.114 | 0.175 | 0.837 | 0.530 |
| 0.750 | True | 0.181 | 0.133 | 0.194 | 0.832 | 0.549 |
| 0.500 | True | 0.213 | 0.162 | 0.224 | 0.799 | 0.540 |
| 0.250 | True | 0.274 | 0.244 | 0.295 | 0.748 | 0.552 |
| 0.000 | True | 0.572 | 0.526 | 0.551 | 0.406 | 0.494 |

- `good_recall`: share of Good Fit pairs the filter keeps (losing them is the costly error).
- `no_fit_removed`: share of No Fit pairs removed (the work the filter saves).
- `precision_of_kept`: share of kept pairs that are Good/Potential; compare with the no-filter baseline.

Rejection reasons at the strict setting (coverage 1.0, role on): must-have: 1390, role: 628, over-qualified: 29, years: 226