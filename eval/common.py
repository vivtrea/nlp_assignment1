"""Shared helpers for eval scripts: metrics, data loading, report writing."""

from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recruiter import config  # noqa: E402

GAIN = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def dcg(gains: list[float]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_gains: list[float], all_gains: list[float], k: int = 10) -> float:
    ideal = dcg(sorted(all_gains, reverse=True)[:k])
    return dcg(ranked_gains[:k]) / ideal if ideal > 0 else 0.0


def mrr(ranked_gains: list[float], min_gain: float = 1) -> float:
    for i, g in enumerate(ranked_gains):
        if g >= min_gain:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_gains: list[float], n_relevant: int, k: int, min_gain: float = 1) -> float:
    return sum(1 for g in ranked_gains[:k] if g >= min_gain) / n_relevant if n_relevant else 0.0


def spearman(a: list[float], b: list[float]) -> float:
    from scipy.stats import spearmanr

    if len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return round(float(np.mean(xs)), 4) if xs else float("nan")


def write_report(component: str, results: dict[str, Any], markdown: str) -> None:
    config.REPORTS_EVAL.mkdir(parents=True, exist_ok=True)
    results = {
        "component": component,
        "date": str(date.today()),
        "llm_enabled": config.llm_enabled(),
        "llm_model": config.LLM_MODEL if config.llm_enabled() else None,
        **results,
    }
    (config.REPORTS_EVAL / f"{component}.json").write_text(json.dumps(results, indent=2, default=str))
    (config.REPORTS_EVAL / f"{component}.md").write_text(markdown)
    print(markdown)


def md_table(rows: list[dict[str, Any]], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(no rows)_"
    cols = cols or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "n/a" if math.isnan(v) else f"{v:.3f}"
    return str(v)


def load_fit(split: str = "test"):
    import pandas as pd

    df = pd.read_parquet(config.DATA_RAW / "fit_labels" / f"{split}.parquet")
    df["gain"] = df["label"].map(GAIN)
    return df
