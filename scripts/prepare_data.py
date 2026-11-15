"""CRISP-DM phase 3 · Data Preparation.

Builds data/processed/candidates.parquet (10k stratified Djinni sample -> CandidateProfile
rows) and data/processed/index/hybrid.pkl (BM25 + LSA index). Writes the numbers used in
reports/03_data_preparation.md to reports/eval/data_preparation.json.

Usage: python scripts/prepare_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recruiter import config  # noqa: E402
from recruiter.c2_profiles import profile_from_djinni  # noqa: E402
from recruiter.c4_retrieval import HybridIndex, candidate_text  # noqa: E402


def stratified_sample(df: pd.DataFrame, n: int, min_per_class: int, seed: int) -> pd.DataFrame:
    counts = df["Primary Keyword"].value_counts()
    alloc = (counts / counts.sum() * n).round().astype(int).clip(lower=min_per_class)
    alloc = pd.concat([alloc, counts], axis=1).min(axis=1).astype(int)
    # shrink the largest classes so the total is exactly n
    while alloc.sum() > n:
        alloc[alloc.idxmax()] -= 1
    parts = [g.sample(alloc[k], random_state=seed) for k, g in df.groupby("Primary Keyword")]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def main() -> None:
    raw = pd.read_parquet(config.DATA_RAW / "djinni_cvs" / "train.parquet")
    stats: dict = {"raw_rows": len(raw)}
    df = raw.drop(columns=["__index_level_0__", "CV_lang"], errors="ignore")
    df = df[df["Primary Keyword"].notna()]
    stats["after_drop_null_keyword"] = len(df)
    df = df[df["CV"].fillna("").str.strip().str.len() >= config.MIN_CV_CHARS]
    stats["after_drop_short_cv"] = len(df)
    df = df.drop_duplicates(subset=["CV"])
    stats["after_dedup_cv"] = len(df)
    df["English Level"] = df["English Level"].fillna("unknown")

    sample = stratified_sample(df, config.SAMPLE_SIZE, config.MIN_PER_CLASS, config.SEED)
    stats["sample_rows"] = len(sample)
    stats["sample_per_keyword"] = sample["Primary Keyword"].value_counts().to_dict()

    profiles = [profile_from_djinni(r) for r in sample.to_dict("records")]
    out = pd.DataFrame([p.model_dump() for p in profiles])
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.CANDIDATES_PATH)
    stats["profiles_with_0_skills"] = int((out["skills"].map(len) == 0).sum())
    stats["skills_per_profile_median"] = float(out["skills"].map(len).median())
    stats["english_unknown"] = int((out["english_level"] == "unknown").sum())
    stats["looking_for_empty"] = int((out["looking_for"] == "").sum())
    stats["seniority"] = out["seniority"].value_counts().to_dict()

    index = HybridIndex.build(out["candidate_id"].tolist(), [candidate_text(t, c) for t, c in zip(out["title"], out["cv_text"])])
    index.save(config.INDEX_DIR / "hybrid.pkl")
    stats["index_vocab"] = len(index.tfidf.vocabulary_)
    stats["lsa_dim"] = int(index.doc_vecs.shape[1])
    stats["lsa_explained_variance"] = round(float(index.svd.explained_variance_ratio_.sum()), 4) if index.svd else None

    config.REPORTS_EVAL.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_EVAL / "data_preparation.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "sample_per_keyword"}, indent=2))


if __name__ == "__main__":
    main()
