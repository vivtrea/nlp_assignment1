# Evaluation summary (2026-11-15)

LLM enabled during the last run: **False**. Every number below is read from `reports/eval/*.json`; per-component details are in `reports/eval/<component>.md`.

| Component | Metric | System | Baseline |
|---|---|---|---|
| C1 collector | completion rate (held-out dialogues) | 0.9 | — |
| C1 collector | must-have skill F1 (held-out / dev) | 0.9286 / 0.9647 | — |
| C1 collector | mean extra turns (held-out) | 0.3 | 0 = optimal |
| C1 single-shot | role accuracy on 200 Djinni JDs | 0.675 | 0.125 (majority) |
| C2 skills | P / R / F1 on known tech skills (500 CVs) | 0.9483 / 0.7203 / 0.8187 | 0.9976 / 0.8273 / 0.9045 (gazetteer) |
| C3 filter | Good-Fit recall / No-Fit removed / precision of kept (strict) | 0.1135 / 0.8366 / 0.5302 | 1.0 / 0.0 / 0.5128 (no filter) |
| C4 retrieval | nDCG@10 / MRR / Recall@50 (pooled) | 0.0955 / 0.2185 / 0.2848 | BM25 0.0892 / 0.1922 / 0.2629; random 0.0311 / 0.1172 / 0.1461 |
| C5 re-ranker | in-pool nDCG@10: rubric only / final blend (see c5 report) | 0.5626 / 0.6887 | hybrid 0.6821, random 0.5682 |
| C5 re-ranker | macro-F1 3-class (test) | 0.3439 | 0.2184 (majority) |
| C6 outreach | faithful messages / CTA / length in range | 1.0 / 1.0 / 1.0 | — |
| C6 verifier | detection recall: fake skill / years / quote / domain | 1.0 / 0.8814 / 1.0 / 1.0 | FP rate 0.0 |
| C6 outreach | same-job trigram Jaccard (lower = more personal) | 0.4586 | — |
| C5 vs manual labels (45 Djinni pairs, 10 jobs) | top-3 precision (Good or Potential) / share Good | 0.9 / 0.4667 | — |
| C5 vs manual labels | Spearman final / rubric only; #1 is Good | 0.4512 / 0.2301; 7/10 | — |
| C3 vs manual labels | labelled candidates rejected by the system | 8 ({'No Fit': 8}) | — |
| C6 vs manual labels | mean message score (1–5), messages of the current top 3 | 3.0 | — |

Manual-label agreement: complete.
