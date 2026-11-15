"""Eval C4 · Hybrid retrieval on fit_labels test.

Pooled setting: each of the 71 JDs is a query against ALL 477 unique test resumes.
Relevance = human label for that (JD, resume) pair: Good=2, Potential=1, unlabelled/No=0.
Unlabelled pairs count as non-relevant, so absolute numbers are a lower bound; the
comparison between methods is what matters. Queries with no relevant resume are skipped.
Also measures C3 -> C4 (filter first, then retrieve), as in the pipeline.
"""

from __future__ import annotations

import random

from common import config, md_table, mean, mrr, ndcg_at_k, recall_at_k, write_report
from fit_data import bed, index
from recruiter import c3_filter

K = 10
R = 50


def evaluate(b, idx, rank_fn, label: str) -> dict:
    nd, mr, rc, p10 = [], [], [], []
    for jd, grp in b.pairs.groupby("jd_id"):
        gains = dict(zip(grp["cv_id"], grp["gain"]))
        n_rel = sum(1 for g in gains.values() if g > 0)
        if n_rel == 0:
            continue
        ranked = rank_fn(jd)
        rg = [gains.get(c, 0) for c in ranked]
        nd.append(ndcg_at_k(rg, list(gains.values()), K))
        mr.append(mrr(rg))
        rc.append(recall_at_k(rg, n_rel, R))
        p10.append(sum(1 for g in rg[:K] if g > 0) / K)
    return {
        "method": label,
        "queries": len(nd),
        f"nDCG@{K}": mean(nd),
        "MRR": mean(mr),
        f"Recall@{R}": mean(rc),
        f"P@{K}": mean(p10),
    }


def main() -> None:
    b = bed("test")
    idx = index("test")
    all_ids = list(idx.ids)
    rnd = random.Random(config.SEED)

    def rand(_jd):
        ids = all_ids[:]
        rnd.shuffle(ids)
        return ids

    def search(mode: str, raw: bool, filtered: float | None = None):
        def fn(jd):
            q = b.jd_text[jd] if raw else b.reqs[jd].query_text()
            restrict = None
            if filtered is not None:
                restrict = [
                    c for c in all_ids if c3_filter.check(b.reqs[jd], b.profiles[c], use_role=True, min_coverage=filtered).passed
                ]
            return [c for c, _ in idx.search(q, top_n=len(all_ids), restrict_to=restrict, mode=mode)]

        return fn

    rows = [evaluate(b, idx, rand, "random (baseline)")]
    for raw in (False, True):
        src = "raw JD text" if raw else "C1 query (requirements)"
        for mode in ("bm25", "dense", "hybrid"):
            name = {"bm25": "BM25", "dense": "LSA dense", "hybrid": "Hybrid RRF"}[mode]
            rows.append(evaluate(b, idx, search(mode, raw), f"{name} · {src}"))
    for cov in (0.5, 0.25):
        rows.append(evaluate(b, idx, search("hybrid", False, cov), f"C3 (coverage {cov}, role) → Hybrid · C1 query"))
    md = "\n".join(
        [
            "# C4 · Hybrid retrieval: evaluation (fit_labels test, pooled)",
            "",
            f"Pool: {len(all_ids)} unique resumes. Gains: Good=2, Potential=1, No/unlabelled=0. "
            f"nDCG@{K} (graded), MRR (first Good/Potential), Recall@{R}, P@{K}. LSA dim = {idx.doc_vecs.shape[1]}.",
            "",
            md_table(rows),
            "",
            "Unlabelled (JD, resume) pairs count as non-relevant, so absolute values are lower bounds; compare methods.",
            "Filtered rows: resumes removed by C3 can't be retrieved, so recall is capped by the filter.",
        ]
    )
    write_report("c4_retrieval", {"rows": rows, "pool": len(all_ids)}, md)


if __name__ == "__main__":
    main()
