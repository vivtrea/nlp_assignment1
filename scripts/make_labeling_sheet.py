"""Create data/eval/labeling_sheet.csv for human labels (C5 fit, C6 message quality).

10 Djinni JDs (seed 7, disjoint from the C6 eval seed) x 3 candidates: rank 1, rank 2 and
one random eligible candidate from the bottom half of the retrieved list, so labels cover
high and low scores. Fill `human_fit` and `human_message_score` following
eval/rubrics/rubric_scorer.md, then run `python eval/eval_human_agreement.py`.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recruiter import c6_outreach, config, pipeline  # noqa: E402
from recruiter.c1_requirements import requirements_from_jd  # noqa: E402

OUT = config.DATA_EVAL / "labeling_sheet.csv"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; delete it first if you really want to regenerate (labels would be lost).")
    jobs = pd.read_parquet(config.DATA_RAW / "djinni_jobs" / "train.parquet").sample(10, random_state=7)
    rnd = random.Random(7)
    rows = []
    for r in jobs.to_dict("records"):
        req = requirements_from_jd(f"{r['Position']}\n{r['Long Description']}")
        req.title = r["Position"]
        res = pipeline.run(req, top_k=config.RETRIEVAL_TOP_N, n_messages=0)
        if len(res.ranked) < 3:
            continue
        low = rnd.choice(res.ranked[len(res.ranked) // 2 :])
        for rc in (res.ranked[0], res.ranked[1], low):
            p = res.profiles[rc.candidate_id]
            msg = c6_outreach.write_message(req, p, rc).text
            rows.append(
                {
                    "job_id": r["id"],
                    "job_title": r["Position"],
                    "requirements": req.model_dump_json(exclude={"summary", "pending_skills"}),
                    "candidate_id": p.candidate_id,
                    "candidate_title": p.title,
                    "years": p.years_experience,
                    "english": p.english_level,
                    "looking_for": p.looking_for,
                    "cv_excerpt": p.cv_text[:800],
                    "message": msg,
                    "human_fit": "",
                    "human_message_score": "",
                    "system_total_HIDE_WHILE_LABELLING": rc.total,
                    "system_final_HIDE_WHILE_LABELLING": rc.final,
                    "scorer": rc.scorer,
                    "system_criteria": "; ".join(f"{c.criterion}={c.score}" for c in rc.criteria),
                }
            )
    pd.DataFrame(rows).sample(frac=1, random_state=7).to_csv(OUT, index=False)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
