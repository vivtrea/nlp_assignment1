"""End-to-end pipeline: JobRequirements -> C3 -> C4 -> C5 -> C6, with a per-stage trace."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from recruiter import c3_filter, c5_reranker, c6_outreach, config
from recruiter.c4_retrieval import HybridIndex
from recruiter.schemas import CandidateProfile, FilterDecision, JobRequirements, OutreachMessage, RankedCandidate


@dataclass
class PipelineResult:
    requirements: JobRequirements
    eligible: int
    total: int
    retrieved: list[tuple[str, float]]
    ranked: list[RankedCandidate]
    messages: list[OutreachMessage]
    profiles: dict[str, CandidateProfile]
    rejections: dict[str, int] = field(default_factory=dict)
    must_have_coverage: float = 1.0
    trace: list[dict] = field(default_factory=list)


def _profile_from_row(row: dict) -> CandidateProfile:
    data = {k: row[k] for k in CandidateProfile.model_fields if k in row}
    for k in ("skills", "domains"):
        data[k] = list(data.get(k) if data.get(k) is not None else [])
    if pd.isna(data.get("years_experience")):
        data["years_experience"] = None
    return CandidateProfile(**data)


@lru_cache(maxsize=1)
def load_store() -> tuple[dict[str, CandidateProfile], HybridIndex]:
    if not config.CANDIDATES_PATH.exists():
        raise FileNotFoundError("Run `python scripts/prepare_data.py` first.")
    df = pd.read_parquet(config.CANDIDATES_PATH)
    profiles = {r["candidate_id"]: _profile_from_row(r) for r in df.to_dict("records")}
    index = HybridIndex.load(config.INDEX_DIR / "hybrid.pkl")
    return profiles, index


def _rejection_summary(decisions: list[FilterDecision]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in decisions:
        for r in d.reasons:
            key = r.split(":")[0].split(" ")[0] if not r.startswith("missing") else "missing must-have"
            key = {"role": "role mismatch", "English": "English too low", "over-qualified": "over-qualified"}.get(key, key)
            if "y < required" in r:
                key = "not enough years"
            out[key] = out.get(key, 0) + 1
    return out


def run(
    req: JobRequirements,
    *,
    top_n: int = config.RETRIEVAL_TOP_N,
    top_k: int = config.RERANK_TOP_K,
    n_messages: int = config.OUTREACH_TOP_K,
    scorer: c5_reranker.RuleScorer | c5_reranker.LLMScorer | None = None,
    writer: c6_outreach.TemplateWriter | c6_outreach.LLMWriter | None = None,
) -> PipelineResult:
    profiles, index = load_store()
    trace = []

    t = time.time()
    coverage = config.FILTER_MUST_HAVE_MIN_COVERAGE
    kept, decisions = c3_filter.apply(req, list(profiles.values()), min_coverage=coverage)
    for relaxed in config.FILTER_RELAXATION_LADDER:
        if len(kept) >= top_k:
            break
        coverage = relaxed
        kept, decisions = c3_filter.apply(req, list(profiles.values()), min_coverage=coverage)
    trace.append(
        {
            "stage": "C3 hard filter",
            "in": len(profiles),
            "out": len(kept),
            "must_have_coverage": coverage,
            "sec": round(time.time() - t, 3),
        }
    )

    t = time.time()
    retrieved = index.search(req.query_text(), top_n=top_n, restrict_to=[p.candidate_id for p in kept])
    trace.append(
        {"stage": "C4 hybrid retrieval (BM25+LSA, RRF)", "in": len(kept), "out": len(retrieved), "sec": round(time.time() - t, 3)}
    )

    t = time.time()
    ret_map = {cid: (i + 1, s) for i, (cid, s) in enumerate(retrieved)}
    ranked = c5_reranker.rerank(req, [profiles[c] for c, _ in retrieved], scorer=scorer, retrieval=ret_map)[:top_k]
    trace.append(
        {
            "stage": f"C5 rubric re-rank (rules on {len(retrieved)}"
            + (f"; LLM on top {n_llm})" if (n_llm := sum(r.scorer == "llm" for r in ranked)) else ")"),
            "in": len(retrieved),
            "out": len(ranked),
            "sec": round(time.time() - t, 3),
        }
    )

    t = time.time()
    msgs = [c6_outreach.write_message(req, profiles[r.candidate_id], r, writer=writer) for r in ranked[:n_messages]]
    trace.append(
        {
            "stage": "C6 outreach + claim verifier",
            "in": len(ranked[:n_messages]),
            "out": len(msgs),
            "faithful": sum(m.faithful for m in msgs),
            "sec": round(time.time() - t, 3),
        }
    )

    return PipelineResult(
        req, len(kept), len(profiles), retrieved, ranked, msgs, profiles, _rejection_summary(decisions), coverage, trace
    )
