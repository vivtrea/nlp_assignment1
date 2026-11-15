"""Central configuration. Rubric weights live ONLY here (CLAUDE.md §5)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EVAL = ROOT / "data" / "eval"
REPORTS = ROOT / "reports"
REPORTS_EVAL = REPORTS / "eval"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

CANDIDATES_PATH = DATA_PROCESSED / "candidates.parquet"
INDEX_DIR = DATA_PROCESSED / "index"
LLM_CACHE_DIR = DATA_PROCESSED / "llm_cache"
LOG_DIR = DATA_PROCESSED / "logs"

SEED = 42
SAMPLE_SIZE = 10_000
MIN_PER_CLASS = 60
MIN_CV_CHARS = 50

# --- C5 rubric (klearskill-style interview evaluation form). Must sum to 1.0. ---
RUBRIC_WEIGHTS: dict[str, float] = {
    "must_have_skills": 0.35,
    "relevant_experience": 0.25,
    "seniority_fit": 0.10,
    "nice_to_have": 0.10,
    "english": 0.10,
    "motivation": 0.10,
}

# --- C3 hard filter ---
FILTER_YEARS_TOLERANCE = 0.0  # candidate years >= min_years - tolerance
FILTER_UNKNOWN_ENGLISH_PASSES = True
FILTER_UNKNOWN_YEARS_PASSES = True
FILTER_MUST_HAVE_MIN_COVERAGE = 1.0  # share of must-have skills a candidate must show
# If fewer than RERANK_TOP_K candidates survive, relax must-have coverage step by step.
# Shown to the recruiter in the trace; set to () to disable.
FILTER_RELAXATION_LADDER: tuple[float, ...] = (0.75, 0.5)
# Over-qualification rule (approved 2026-11-15): for jobs at these levels, remove candidates
# whose seniority is >= FILTER_OVERQUALIFIED_GAP levels above (junior job -> no senior/lead).
FILTER_OVERQUALIFIED_FOR: tuple[str, ...] = ("junior",)
FILTER_OVERQUALIFIED_GAP = 2
# One-shot JD extraction keeps at most this many must-haves (title skills first, then
# requirement sections); the rest become nice-to-have. Manual labels showed 6-12-item lists
# that no real candidate matches.
ONESHOT_MAX_MUST = 6

# --- C4 retrieval ---
RETRIEVAL_TOP_N = 50
RRF_K = 60
LSA_DIM = 256
DENSE_BACKEND = os.getenv("DENSE_BACKEND", "lsa")  # "lsa" (offline) — pluggable

# --- C5 / C6 ---
RERANK_TOP_K = 10
# Final order = alpha * rubric total + (1 - alpha) * hybrid retrieval score, both min-max
# normalised per query. Tuned on fit_labels TRAIN (eval_c5_reranker.py); approved 2026-11-15.
RERANK_BLEND_ALPHA = 0.1
# Two-stage C5: rules score all RETRIEVAL_TOP_N candidates, the LLM re-scores only the best
# LLM_RERANK_TOP of them (keeps a search at ~17 LLM calls; free tier = 50 requests/day).
LLM_RERANK_TOP = 10
OUTREACH_TOP_K = 3
OUTREACH_MIN_CHARS = 250
OUTREACH_MAX_CHARS = 1000  # LinkedIn InMail is 1,900; recruiters keep it short

ENGLISH_ORDER = ["no_english", "basic", "pre", "intermediate", "upper", "fluent"]
SENIORITY_ORDER = ["junior", "middle", "senior", "lead"]


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

LLM_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "baidu/ernie-5.5-flash-preview:free")
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or LLM_MODEL
TEMPERATURE_SCORING = 0.0
TEMPERATURE_WRITING = 0.7


def llm_enabled() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


assert abs(sum(RUBRIC_WEIGHTS.values()) - 1.0) < 1e-9, "rubric weights must sum to 1"
