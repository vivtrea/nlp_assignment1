# C4 · Hybrid retrieval: evaluation (fit_labels test, pooled)

Pool: 477 unique resumes. Gains: Good=2, Potential=1, No/unlabelled=0. nDCG@10 (graded), MRR (first Good/Potential), Recall@50, P@10. LSA dim = 128.

| method | queries | nDCG@10 | MRR | Recall@50 | P@10 |
|---|---|---|---|---|---|
| random (baseline) | 61 | 0.031 | 0.117 | 0.146 | 0.023 |
| BM25 · C1 query (requirements) | 61 | 0.089 | 0.192 | 0.263 | 0.093 |
| LSA dense · C1 query (requirements) | 61 | 0.086 | 0.197 | 0.315 | 0.093 |
| Hybrid RRF · C1 query (requirements) | 61 | 0.096 | 0.218 | 0.285 | 0.093 |
| BM25 · raw JD text | 61 | 0.082 | 0.160 | 0.241 | 0.087 |
| LSA dense · raw JD text | 61 | 0.092 | 0.210 | 0.286 | 0.100 |
| Hybrid RRF · raw JD text | 61 | 0.098 | 0.228 | 0.271 | 0.095 |
| C3 (coverage 0.5, role) → Hybrid · C1 query | 61 | 0.086 | 0.209 | 0.157 | 0.080 |
| C3 (coverage 0.25, role) → Hybrid · C1 query | 61 | 0.092 | 0.213 | 0.188 | 0.092 |

Unlabelled (JD, resume) pairs count as non-relevant, so absolute values are lower bounds; compare methods.
Filtered rows: resumes removed by C3 can't be retrieved, so recall is capped by the filter.