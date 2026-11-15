"""Agreement between system scores and manual labels (data/eval/labeling_sheet.csv).

Run scripts/rescore_labeling_sheet.py first after any code change so system columns are current.

C5: Spearman(system total, human fit) and macro-F1 after mapping totals with the C5
    thresholds. C6: mean human message score; if an LLM judge is configured, judge-vs-human
    Spearman (CLAUDE.md §7 requires >= 20 items).
"""

from __future__ import annotations

import pandas as pd

from common import config, md_table, spearman, write_report

GAIN = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}


def main() -> None:
    path = config.DATA_EVAL / "labeling_sheet.csv"
    if not path.exists():
        write_report(
            "human_agreement",
            {"status": "no sheet"},
            "# Human agreement\n\n_Pending: run `python scripts/make_labeling_sheet.py` and fill the labels._\n",
        )
        return
    df = pd.read_csv(path)
    lab = df[df["human_fit"].isin(GAIN)].copy()
    lab["y"] = lab["human_fit"].map(GAIN)
    res: dict = {"rows": len(df), "fit_labelled": len(lab), "human_label_counts": lab["human_fit"].value_counts().to_dict()}
    if len(lab) >= 5:
        res["c5_spearman_final_vs_human"] = round(
            spearman(lab["system_final_HIDE_WHILE_LABELLING"].tolist(), lab["y"].tolist()), 4
        )
        res["c5_spearman_total_vs_human"] = round(
            spearman(lab["system_total_HIDE_WHILE_LABELLING"].tolist(), lab["y"].tolist()), 4
        )
        if "system_rank" in lab:
            top3 = lab[lab["system_rank"].fillna(99) <= 3]
            jobs = df["job_id"].nunique()
            res["top3_slots_labelled"] = f"{len(top3)}/{3 * jobs}"
            res["top3_precision_good_or_potential"] = round(float((top3["y"] > 0).mean()), 4)
            res["top3_share_good"] = round(float((top3["y"] == 2).mean()), 4)
            top1 = lab[lab["system_rank"] == 1]
            res["top1_good"] = f"{int((top1['y'] == 2).sum())}/{len(top1)}"
            res["top1_good_or_potential"] = f"{int((top1['y'] > 0).sum())}/{len(top1)}"
            best = lab.groupby("job_id")["y"].max()
            res["top1_is_best_labelled"] = f"{int(sum(r.y == best[r.job_id] for r in top1.itertuples()))}/{len(top1)}"
            rej = lab[~lab["system_retrieved"].astype(bool)]
            res["labelled_rejected_by_system"] = len(rej)
            res["labels_of_rejected"] = rej["human_fit"].value_counts().to_dict()
        if "label_round" in lab:
            r1 = lab[lab["label_round"] == 1]
            res["round1_rows"] = len(r1)
            # like-for-like: the same first-round pairs, scored by the current code
            res["round1_spearman_final_current_code"] = round(
                spearman(r1["system_final_HIDE_WHILE_LABELLING"].tolist(), r1["y"].tolist()), 4
            )
            kept = r1[r1["system_retrieved"].astype(bool)]
            res["round1_spearman_final_current_code_retrieved_only"] = round(
                spearman(kept["system_final_HIDE_WHILE_LABELLING"].tolist(), kept["y"].tolist()), 4
            )
            res["round1_retrieved"] = f"{len(kept)}/{len(r1)}"
            # recorded from the first agreement run (code before the C1/C3/C6 fixes), not recomputable now
            res["round1_spearman_final_old_code"] = 0.141
            res["round1_top1_is_best_labelled_old_code"] = "8/10"
    for col, key in (
        ("human_message_score", "c6_mean_message_score_v1"),
        ("human_message_score_v2", "c6_mean_message_score_current"),
    ):
        if col in df:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v):
                res[key] = round(float(v.mean()), 3)
    if "system_rank" in df and "human_message_score_v2" in df:
        sent = pd.to_numeric(df.loc[df["system_rank"].fillna(99) <= 3, "human_message_score_v2"], errors="coerce").dropna()
        if len(sent):
            res["c6_mean_message_score_top3_sent"] = round(float(sent.mean()), 3)
    status = (
        "complete" if len(lab) >= 20 and not df["human_fit"].isna().any() else "pending (need >= 20 labelled rows, none empty)"
    )
    res["status"] = status
    md = "\n".join(
        [
            "# Agreement with manual labels (C5 ranking, C6 messages)",
            "",
            f"Status: **{status}**",
            "",
            md_table([{k: str(v) for k, v in res.items()}], ["metric", "value"])
            if False
            else "\n".join(["| metric | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in res.items()]),
        ]
    )
    write_report("human_agreement", res, md)


if __name__ == "__main__":
    main()
