# C5 · Rubric re-ranker: evaluation (fit_labels)

## In-pool re-ranking (each test JD's labelled resumes)

| method | JDs | nDCG@10 | Spearman (per-JD mean) |
|---|---|---|---|
| random (baseline) | 61 | 0.568 | 0.012 |
| BM25 score (C4 sparse) | 61 | 0.659 | 0.199 |
| Hybrid RRF score (C4) | 61 | 0.682 | 0.232 |
| Rubric · rules (C5) | 61 | 0.563 | 0.056 |
| Rubric · rules + hybrid tie-break (pipeline before 2026-11-15) | 61 | 0.598 | 0.088 |
| PIPELINE (C5 final): 0.1·rubric + 0.9·hybrid (α from config; best α on train = 0.1) | 61 | 0.689 | 0.230 |

Blend experiment, train nDCG@10 by alpha (weight of the rubric): 0.0: 0.7321, 0.1: 0.7351, 0.2: 0.7257, 0.3: 0.7153, 0.4: 0.7048, 0.5: 0.6788, 0.6: 0.6565, 0.7: 0.6398, 0.8: 0.6298, 0.9: 0.6218, 1.0: 0.5565

## Score → 3-class fit label

Thresholds tuned on train (macro-F1 0.3423): No < 0.1625 ≤ Potential < 0.275 ≤ Good.

| system | macro-F1 (test) | global Spearman |
|---|---|---|
| Rubric · rules | 0.344 | -0.009 |
| majority class (No Fit) | 0.218 | n/a |

Mean rubric total by human label: No Fit: 0.1967, Potential Fit: 0.2112, Good Fit: 0.1894

## Which criteria carry signal?

| criterion | weight | Spearman vs label | share scored 1 | mean |
|---|---|---|---|---|
| must_have_skills | 0.350 | 0.015 | 0.390 | 2.147 |
| relevant_experience | 0.250 | -0.014 | 0.775 | 1.644 |
| seniority_fit | 0.100 | -0.014 | 0.637 | 1.875 |
| nice_to_have | 0.100 | 0.019 | 0.449 | 2.361 |
| english | 0.100 | -0.008 | 0.980 | 1.077 |
| motivation | 0.100 | n/a | 1.000 | 1.000 |

## LLM scorer

_Not run: no OPENROUTER_API_KEY. Run `python eval/eval_c5_reranker.py` after adding a key._