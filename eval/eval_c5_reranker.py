"""Eval C5 · Rubric re-ranker on fit_labels.

In-pool re-ranking: for each test JD, rank only its labelled resumes (median 20).
Compares rubric total vs retrieval scores vs random: nDCG@10, per-JD Spearman.
Classification view: two thresholds on the rubric total (tuned on TRAIN for macro-F1)
map scores to No/Potential/Good Fit; macro-F1 on TEST vs a majority-class baseline.
Per-criterion Spearman shows which rubric criteria carry signal.
LLM scorer runs on a capped subset when an API key is set.
"""

from __future__ import annotations

import random
from itertools import product

import numpy as np
from sklearn.metrics import f1_score

from common import config, md_table, mean, ndcg_at_k, spearman, write_report
from fit_data import bed, index
from recruiter import llm
from recruiter.c5_reranker import CRITERIA, LLMScorer, RuleScorer, total_score

LABELS = ["No Fit", "Potential Fit", "Good Fit"]
LLM_MAX_PAIRS = 100


def rubric_scores(b, scorer) -> dict[tuple[str, str], list]:
    out = {}
    for r in b.pairs.itertuples():
        out[(r.jd_id, r.cv_id)] = scorer.score(b.reqs[r.jd_id], b.profiles[r.cv_id])
    return out


def to_class(score: float, t1: float, t2: float) -> str:
    return LABELS[0] if score < t1 else LABELS[1] if score < t2 else LABELS[2]


def tune_thresholds(scores: list[float], labels: list[str]) -> tuple[float, float, float]:
    grid = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 37)))
    best = (0.0, 0.0, 1.0)
    for t1, t2 in product(grid, grid):
        if t2 <= t1:
            continue
        f = f1_score(labels, [to_class(s, t1, t2) for s in scores], average="macro", labels=LABELS)
        if f > best[0]:
            best = (f, float(t1), float(t2))
    return best


def ranking_metrics(b, score_of, name: str) -> dict:
    nd, sp = [], []
    for jd, grp in b.pairs.groupby("jd_id"):
        if grp["gain"].nunique() < 2:
            continue
        s = [score_of(jd, c) for c in grp["cv_id"]]
        g = grp["gain"].tolist()
        order = np.argsort(-np.array(s), kind="stable")
        nd.append(ndcg_at_k([g[i] for i in order], g, 10))
        sp.append(spearman(s, g))
    return {"method": name, "JDs": len(nd), "nDCG@10": mean(nd), "Spearman (per-JD mean)": mean(sp)}


def main() -> None:
    test, train = bed("test"), bed("train")
    idx = index("test")
    rule_test = rubric_scores(test, RuleScorer())
    rule_train = rubric_scores(train, RuleScorer())

    retr: dict[tuple[str, str], float] = {}
    for mode in ("bm25", "hybrid"):
        for jd, grp in test.pairs.groupby("jd_id"):
            res = dict(idx.search(test.reqs[jd].query_text(), top_n=len(grp), restrict_to=grp["cv_id"].tolist(), mode=mode))
            for c in grp["cv_id"]:
                retr[(mode, jd, c)] = res.get(c, 0.0)

    # experiment: blend per-JD min-max normalised rubric and hybrid scores, alpha tuned on TRAIN
    def hybrid_scores(b, ix):
        out = {}
        for jd, grp in b.pairs.groupby("jd_id"):
            res = dict(ix.search(b.reqs[jd].query_text(), top_n=len(grp), restrict_to=grp["cv_id"].tolist()))
            out.update({(jd, c): res.get(c, 0.0) for c in grp["cv_id"]})
        return out

    def blend_fn(b, rub, hyb, alpha):
        norm = {}
        for jd, grp in b.pairs.groupby("jd_id"):
            for src, tag in ((rub, "r"), (hyb, "h")):
                v = {c: (total_score(src[(jd, c)]) if tag == "r" else src[(jd, c)]) for c in grp["cv_id"]}
                lo, hi = min(v.values()), max(v.values())
                for c, x in v.items():
                    norm[(tag, jd, c)] = (x - lo) / (hi - lo) if hi > lo else 0.5
        return lambda j, c: alpha * norm[("r", j, c)] + (1 - alpha) * norm[("h", j, c)]

    hyb_train = hybrid_scores(train, index("train"))
    hyb_test = {(j, c): retr[("hybrid", j, c)] for (j, c) in rule_test}
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_train = {a: ranking_metrics(train, blend_fn(train, rule_train, hyb_train, a), "")["nDCG@10"] for a in alphas}
    best_alpha = max(alpha_train, key=alpha_train.get)

    rnd = random.Random(config.SEED)
    rand = {k: rnd.random() for k in rule_test}
    rows = [
        ranking_metrics(test, lambda j, c: rand[(j, c)], "random (baseline)"),
        ranking_metrics(test, lambda j, c: retr[("bm25", j, c)], "BM25 score (C4 sparse)"),
        ranking_metrics(test, lambda j, c: retr[("hybrid", j, c)], "Hybrid RRF score (C4)"),
        ranking_metrics(test, lambda j, c: total_score(rule_test[(j, c)]), "Rubric · rules (C5)"),
        ranking_metrics(
            test,
            lambda j, c: total_score(rule_test[(j, c)]) + 1e-3 * retr[("hybrid", j, c)],
            "Rubric · rules + hybrid tie-break (pipeline before 2026-11-15)",
        ),
        ranking_metrics(
            test,
            blend_fn(test, rule_test, hyb_test, config.RERANK_BLEND_ALPHA),
            f"PIPELINE (C5 final): {config.RERANK_BLEND_ALPHA}·rubric + {1 - config.RERANK_BLEND_ALPHA:.1f}·hybrid "
            f"(α from config; best α on train = {best_alpha})",
        ),
    ]

    tr_scores = [total_score(rule_train[(r.jd_id, r.cv_id)]) for r in train.pairs.itertuples()]
    f_train, t1, t2 = tune_thresholds(tr_scores, train.pairs["label"].tolist())
    te_scores = [total_score(rule_test[(r.jd_id, r.cv_id)]) for r in test.pairs.itertuples()]
    te_labels = test.pairs["label"].tolist()
    pred = [to_class(s, t1, t2) for s in te_scores]
    cls = {
        "rubric_rules_macro_f1": round(f1_score(te_labels, pred, average="macro", labels=LABELS), 4),
        "majority_macro_f1": round(f1_score(te_labels, ["No Fit"] * len(te_labels), average="macro", labels=LABELS), 4),
        "train_macro_f1": round(f_train, 4),
        "thresholds": [round(t1, 4), round(t2, 4)],
        "global_spearman": round(spearman(te_scores, test.pairs["gain"].tolist()), 4),
        "mean_score_by_label": {
            lab: round(float(np.mean([s for s, x in zip(te_scores, te_labels) if x == lab])), 4) for lab in LABELS
        },
    }
    per_crit = []
    for i, c in enumerate(CRITERIA):
        vals = [rule_test[(r.jd_id, r.cv_id)][i].score for r in test.pairs.itertuples()]
        per_crit.append(
            {
                "criterion": c,
                "weight": config.RUBRIC_WEIGHTS[c],
                "Spearman vs label": round(spearman(vals, test.pairs["gain"].tolist()), 4),
                "share scored 1": round(sum(v == 1 for v in vals) / len(vals), 4),
                "mean": round(float(np.mean(vals)), 3),
            }
        )

    llm_res = None
    if config.llm_enabled():
        sub = test.pairs.sample(min(LLM_MAX_PAIRS, len(test.pairs)), random_state=config.SEED)
        scorer, ls, rs, lab = LLMScorer(), [], [], []
        llm.reset_stats()
        for r in sub.itertuples():
            ls.append(total_score(scorer.score(test.reqs[r.jd_id], test.profiles[r.cv_id])))
            rs.append(total_score(rule_test[(r.jd_id, r.cv_id)]))
            lab.append(r.gain)
        llm_res = {
            "pairs": len(sub),
            "spearman_llm": round(spearman(ls, lab), 4),
            "spearman_rules_same_pairs": round(spearman(rs, lab), 4),
            "llm_fallback_rate": round(llm.STATS["fallbacks"] / max(1, llm.STATS["fallbacks"] + llm.STATS["llm_ok"]), 4),
        }

    md = "\n".join(
        [
            "# C5 · Rubric re-ranker: evaluation (fit_labels)",
            "",
            "## In-pool re-ranking (each test JD's labelled resumes)",
            "",
            md_table(rows),
            "",
            "Blend experiment, train nDCG@10 by alpha (weight of the rubric): "
            + ", ".join(f"{a}: {v}" for a, v in alpha_train.items()),
            "",
            "## Score → 3-class fit label",
            "",
            f"Thresholds tuned on train (macro-F1 {cls['train_macro_f1']}): No < {cls['thresholds'][0]} ≤ Potential < {cls['thresholds'][1]} ≤ Good.",
            "",
            md_table(
                [
                    {
                        "system": "Rubric · rules",
                        "macro-F1 (test)": cls["rubric_rules_macro_f1"],
                        "global Spearman": cls["global_spearman"],
                    },
                    {
                        "system": "majority class (No Fit)",
                        "macro-F1 (test)": cls["majority_macro_f1"],
                        "global Spearman": float("nan"),
                    },
                ]
            ),
            "",
            "Mean rubric total by human label: " + ", ".join(f"{k}: {v}" for k, v in cls["mean_score_by_label"].items()),
            "",
            "## Which criteria carry signal?",
            "",
            md_table(per_crit),
            "",
            "## LLM scorer",
            "",
            (
                md_table([llm_res])
                if llm_res
                else "_Not run: no OPENROUTER_API_KEY. Run `python eval/eval_c5_reranker.py` after adding a key._"
            ),
        ]
    )
    write_report(
        "c5_reranker",
        {
            "blend_alpha_train": alpha_train,
            "best_alpha": best_alpha,
            "ranking": rows,
            "classification": cls,
            "per_criterion": per_crit,
            "llm": llm_res,
        },
        md,
    )


if __name__ == "__main__":
    main()
