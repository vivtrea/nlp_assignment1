"""Eval C1 · Requirement collector.

A) 30 hand-written recruiter dialogues (data/eval/c1_dialogues.jsonl) run through the
   slot-filling loop with a simulated recruiter. Metrics: per-slot accuracy, skill
   P/R/F1, completion rate, turns-to-complete vs the minimum possible.
B) 200 Djinni job descriptions (seed 42): single-shot extraction vs the job form fields
   (Primary Keyword, Exp Years, English Level), against a majority-class baseline.
"""

from __future__ import annotations

import json

import pandas as pd

from common import config, md_table, prf, write_report
from recruiter import lexicon, llm
from recruiter.c1_requirements import LLMExtractor, RequirementCollector, RuleExtractor, requirements_from_jd

MAX_TURNS = 8
LLM_JOBS = 50  # free-tier rate limits: smaller sample for the LLM extractor
EXP_MAP = {"no_exp": 0.0, "1y": 1.0, "2y": 2.0, "3y": 3.0, "5y": 5.0}


def simulate(dialogue: dict, extractor) -> tuple[RequirementCollector, int]:
    c = RequirementCollector(extractor)
    c.respond(dialogue["opening"])
    turns = 1
    while not c.done and turns < MAX_TURNS:
        ans = dialogue["answers"].get(c.asked, "I'm not sure, you decide.")
        c.respond(ans)
        turns += 1
    return c, turns


def eval_dialogues(extractor, name: str, file: str = "c1_dialogues.jsonl") -> dict:
    rows = [json.loads(line) for line in (config.DATA_EVAL / file).read_text().splitlines() if line]
    scalar = {k: 0 for k in ("role", "min_years", "english_level", "seniority")}
    counts = {"must": [0, 0, 0], "nice": [0, 0, 0], "all": [0, 0, 0]}
    completed, turns, overhead, errors = 0, [], [], []
    for d in rows:
        c, t = simulate(d, extractor)
        s, g = c.state, d["gold"]
        completed += c.done
        turns.append(t)
        overhead.append(t - (1 + len(d["answers"])))
        for k in scalar:
            got = getattr(s, k)
            ok = (float(got) == float(g[k])) if k == "min_years" and got is not None else got == g[k]
            scalar[k] += ok
            if not ok:
                errors.append({"id": d["id"], "slot": k, "gold": g[k], "pred": got})
        for key, pred, gold in (
            ("must", s.must_have_skills or [], g["must_have_skills"]),
            ("nice", s.nice_to_have_skills, g["nice_to_have_skills"]),
            ("all", (s.must_have_skills or []) + s.nice_to_have_skills, g["must_have_skills"] + g["nice_to_have_skills"]),
        ):
            p, gg = set(pred), set(gold)
            counts[key][0] += len(p & gg)
            counts[key][1] += len(p - gg)
            counts[key][2] += len(gg - p)
            if key != "all" and p != gg:
                errors.append({"id": d["id"], "slot": key, "gold": sorted(gg), "pred": sorted(p)})
    n = len(rows)
    return {
        "extractor": name,
        "set": "dev (used for tuning)" if file == "c1_dialogues.jsonl" else "held-out",
        "n_dialogues": n,
        "completion_rate": round(completed / n, 4),
        "mean_turns": round(sum(turns) / n, 3),
        "mean_extra_turns": round(sum(overhead) / n, 3),
        **{f"acc_{k}": round(v / n, 4) for k, v in scalar.items()},
        "must_have": prf(*counts["must"]),
        "nice_to_have": prf(*counts["nice"]),
        "skills_any_bucket": prf(*counts["all"]),
        "errors": errors,
    }


def eval_djinni_jobs(extractor, name: str, n: int = 200) -> dict:
    jobs = pd.read_parquet(config.DATA_RAW / "djinni_jobs" / "train.parquet").sample(n, random_state=config.SEED)
    role_ok = role_rel = yrs_ok = yrs_close = eng_ok = eng_n = eng_ok_stated = 0
    yrs_err = []
    role_cov = yrs_cov = eng_cov = 0
    for r in jobs.to_dict("records"):
        text = f"{r['Position']}\n{r['Long Description']}"
        req = requirements_from_jd(text, extractor)
        role_cov += req.role is not None
        role_ok += req.role == r["Primary Keyword"]
        role_rel += req.role is not None and r["Primary Keyword"] in lexicon.related_roles(req.role)
        gold_y = EXP_MAP.get(r["Exp Years"])
        raw = extractor.extract(text, req.model_copy(update={"min_years": None}), None).min_years
        yrs_cov += raw is not None
        if gold_y is not None:
            y = raw if raw is not None else 0.0
            yrs_ok += y == gold_y
            yrs_close += abs(y - gold_y) <= 1
            yrs_err.append(abs(y - gold_y))
        if r["English Level"]:
            eng_n += 1
            eng_cov += req.english_level != "no_english"
            eng_ok += req.english_level == r["English Level"]
            eng_ok_stated += req.english_level != "no_english" and req.english_level == r["English Level"]
    maj = {
        "role": jobs["Primary Keyword"].value_counts(normalize=True).iloc[0],
        "years": jobs["Exp Years"].value_counts(normalize=True).iloc[0],
        "english": jobs["English Level"].dropna().value_counts(normalize=True).iloc[0],
    }
    return {
        "extractor": name,
        "n_jobs": n,
        "role_acc": round(role_ok / n, 4),
        "role_acc_related": round(role_rel / n, 4),
        "role_coverage": round(role_cov / n, 4),
        "years_exact": round(yrs_ok / n, 4),
        "years_within_1": round(yrs_close / n, 4),
        "years_mae": round(sum(yrs_err) / len(yrs_err), 3),
        "years_coverage": round(yrs_cov / n, 4),
        "english_acc": round(eng_ok / eng_n, 4),
        "english_coverage": round(eng_cov / eng_n, 4),
        "english_acc_when_stated": round(eng_ok_stated / eng_cov, 4) if eng_cov else None,
        "majority_baseline": {k: round(float(v), 4) for k, v in maj.items()},
    }


def main() -> None:
    variants = [("rules (baseline)", RuleExtractor())]
    if config.llm_enabled():
        variants.append(("llm", LLMExtractor()))

    def tracked(fn, name, *args, **kw):
        llm.reset_stats()
        out = fn(*args, **kw)
        tot = llm.STATS["llm_ok"] + llm.STATS["fallbacks"]
        out["llm_fallback_rate"] = round(llm.STATS["fallbacks"] / tot, 4) if name == "llm" and tot else float("nan")
        return out

    dialogues = [
        tracked(eval_dialogues, n, e, n, f) for f in ("c1_dialogues.jsonl", "c1_dialogues_heldout.jsonl") for n, e in variants
    ]
    jobs = [tracked(eval_djinni_jobs, n, e, n, n=200 if n.startswith("rules") else LLM_JOBS) for n, e in variants]

    drows = [
        {
            "extractor": d["extractor"],
            "set": d["set"],
            "completion": d["completion_rate"],
            "turns": d["mean_turns"],
            "extra turns": d["mean_extra_turns"],
            "role acc": d["acc_role"],
            "years acc": d["acc_min_years"],
            "English acc": d["acc_english_level"],
            "seniority acc": d["acc_seniority"],
            "must-have F1": d["must_have"]["f1"],
            "nice F1": d["nice_to_have"]["f1"],
            "skills F1 (any)": d["skills_any_bucket"]["f1"],
            "LLM fallback rate": d["llm_fallback_rate"],
        }
        for d in dialogues
    ]
    jrows = [
        {
            "extractor": j["extractor"],
            "role acc": j["role_acc"],
            "role acc (related)": j["role_acc_related"],
            "years exact": j["years_exact"],
            "years ±1": j["years_within_1"],
            "years MAE": j["years_mae"],
            "English acc": j["english_acc"],
            "English stated in text": j["english_coverage"],
            "English acc when stated": j["english_acc_when_stated"],
        }
        for j in jobs
    ]
    maj = jobs[0]["majority_baseline"]
    jrows.append(
        {
            "extractor": "majority class",
            "role acc": maj["role"],
            "role acc (related)": "",
            "years exact": maj["years"],
            "years ±1": "",
            "years MAE": "",
            "English acc": maj["english"],
            "English stated in text": "",
        }
    )
    errs = [e | {"set": d["set"][:8]} for d in dialogues if d["extractor"].startswith("rules") for e in d["errors"]]
    md = "\n".join(
        [
            "# C1 · Requirement collector: evaluation",
            "",
            f"LLM enabled: **{config.llm_enabled()}**. Without an API key only the rule-based extractor runs.",
            "",
            "## A) Slot-filling dialogues (30 hand-written, simulated recruiter)",
            "",
            md_table(drows),
            "",
            "`extra turns` = turns used minus the minimum possible (opening + one answer per question the gold script expects).",
            "",
            f"### Errors of the rule extractor ({len(errs)})",
            "",
            md_table([{**e, "gold": str(e["gold"]), "pred": str(e["pred"])} for e in errs[:40]]),
            "",
            "## B) Single-shot extraction from 200 Djinni job descriptions vs. form fields",
            "",
            md_table(jrows),
            "",
            "Gold = the structured fields the employer filled in on Djinni (independent of the text).",
            "`English stated in text` = share of JDs where an English level could be extracted at all.",
        ]
    )
    write_report("c1_requirements", {"dialogues": dialogues, "djinni_jobs": jobs}, md)


if __name__ == "__main__":
    main()
