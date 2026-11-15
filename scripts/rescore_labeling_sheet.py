"""Re-score the manually labelled pairs in data/eval/labeling_sheet.csv with the CURRENT system.

The `human_*` columns are never modified. For each job the current pipeline is re-run (rule
scorer, top 50) and every row gets:
- system_total_HIDE_WHILE_LABELLING: current rubric total for (requirements, candidate)
- system_final_HIDE_WHILE_LABELLING: current blended score in that job's search, or 0.0 if the
  candidate is no longer retrieved (removed by C3 or outside the top 50)
- system_rank: current rank in that job's search (empty if not retrieved)
- message_v2: the message the current writer produces (`human_message_score_v2` refers to it
  for rows labelled before the last template change; `human_message_score` to `message`)
Candidates in a job's CURRENT top 3 that are not in the sheet are appended with empty labels
(label_round = max + 1), so the top-3 precision can be completed by labelling them.

Usage: python scripts/rescore_labeling_sheet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recruiter import c5_reranker, c6_outreach, config, pipeline  # noqa: E402
from recruiter.c1_requirements import requirements_from_jd  # noqa: E402
from recruiter.c5_reranker import RuleScorer  # noqa: E402

SHEET = config.DATA_EVAL / "labeling_sheet.csv"


def main() -> None:
    df = pd.read_csv(SHEET)
    jobs = pd.read_parquet(config.DATA_RAW / "djinni_jobs" / "train.parquet")
    jobs = {r["id"]: r for r in jobs[jobs["id"].isin(set(df["job_id"]))].to_dict("records")}
    profiles, _ = pipeline.load_store()
    next_round = int(df["label_round"].max()) + 1 if "label_round" in df else 2

    searches = {}
    for job_id, j in jobs.items():
        req = requirements_from_jd(f"{j['Position']}\n{j['Long Description']}")
        req.title = j["Position"]
        res = pipeline.run(req, top_k=config.RETRIEVAL_TOP_N, n_messages=0, scorer=RuleScorer())
        searches[job_id] = (req, res.ranked)

    new_rows = []
    for job_id, (_req, ranked) in searches.items():
        have = set(df.loc[df["job_id"] == job_id, "candidate_id"])
        for r in ranked[:3]:
            if r.candidate_id not in have:
                p = profiles[r.candidate_id]
                new_rows.append(
                    {
                        "job_id": job_id,
                        "job_title": jobs[job_id]["Position"],
                        "candidate_id": p.candidate_id,
                        "candidate_title": p.title,
                        "years": p.years_experience,
                        "english": p.english_level,
                        "looking_for": p.looking_for,
                        "cv_excerpt": p.cv_text[:800],
                        "human_fit": "",
                        "human_message_score": "",
                        "human_message_score_v2": "",
                        "label_round": next_round,
                    }
                )
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    cols = {
        k: []
        for k in (
            "requirements",
            "system_total_HIDE_WHILE_LABELLING",
            "system_final_HIDE_WHILE_LABELLING",
            "system_rank",
            "system_retrieved",
            "message_v2",
        )
    }
    for row in df.itertuples():
        req, ranked = searches[row.job_id]
        pos = {r.candidate_id: (i + 1, r) for i, r in enumerate(ranked)}
        p = profiles[row.candidate_id]
        rank, rc = pos.get(row.candidate_id, (None, None))
        rc = rc or c5_reranker.rerank(req, [p], scorer=RuleScorer())[0]
        cols["requirements"].append(req.model_dump_json(exclude={"summary", "pending_skills"}))
        cols["system_total_HIDE_WHILE_LABELLING"].append(rc.total)
        cols["system_final_HIDE_WHILE_LABELLING"].append(rc.final if rank else 0.0)
        cols["system_rank"].append(rank)
        cols["system_retrieved"].append(rank is not None)
        cols["message_v2"].append(c6_outreach.write_message(req, p, rc).text)
    for k, v in cols.items():
        df[k] = v
    # rows added in this run: the message they will be labelled on is the current one
    df.loc[df["message"].isna(), "message"] = df.loc[df["message"].isna(), "message_v2"]
    df.to_csv(SHEET, index=False)
    unl = df[~df["human_fit"].isin(["Good Fit", "Potential Fit", "No Fit"])]
    print(f"rescored {len(df)} rows; {int(df['system_retrieved'].sum())} retrieved; {len(unl)} unlabelled (new top-3)")


if __name__ == "__main__":
    main()
