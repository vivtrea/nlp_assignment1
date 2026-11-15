"""Run every component eval and regenerate reports/eval_summary.md (CLAUDE.md §7).

Usage: python eval/run_all.py [--only c1 c3 ...]
Numbers in the summary are read from reports/eval/*.json; nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVALS = {
    "c1": "eval_c1_requirements.py",
    "c2": "eval_c2_profiles.py",
    "c3": "eval_c3_filter.py",
    "c4": "eval_c4_retrieval.py",
    "c5": "eval_c5_reranker.py",
    "c6": "eval_c6_outreach.py",
    "human": "eval_human_agreement.py",
}


def load(name: str) -> dict:
    p = ROOT / "reports" / "eval" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def summary() -> str:
    c1, c2, c3, c4, c5, c6, hu = (
        load(n)
        for n in ("c1_requirements", "c2_profiles", "c3_filter", "c4_retrieval", "c5_reranker", "c6_outreach", "human_agreement")
    )
    lines = [
        f"# Evaluation summary ({date.today()})",
        "",
        f"LLM enabled during the last run: **{c1.get('llm_enabled')}**. Every number below is read from "
        "`reports/eval/*.json`; per-component details are in `reports/eval/<component>.md`.",
        "",
        "| Component | Metric | System | Baseline |",
        "|---|---|---|---|",
    ]
    if c1:
        dev, held = c1["dialogues"][0], next(d for d in c1["dialogues"] if d["set"] == "held-out")
        j = c1["djinni_jobs"][0]
        lines += [
            f"| C1 collector | completion rate (held-out dialogues) | {held['completion_rate']} | — |",
            f"| C1 collector | must-have skill F1 (held-out / dev) | {held['must_have']['f1']} / {dev['must_have']['f1']} | — |",
            f"| C1 collector | mean extra turns (held-out) | {held['mean_extra_turns']} | 0 = optimal |",
            f"| C1 single-shot | role accuracy on 200 Djinni JDs | {j['role_acc']} | {j['majority_baseline']['role']} (majority) |",
        ]
    if c2:
        rows = {(r["system"], r["gold"]): r for r in c2["ner"]["rows"]}
        lx, gz = rows[("lexicon (ours)", "known")], rows[("gazetteer from other 4,529 gold CVs (baseline)", "known")]
        lines.append(
            f"| C2 skills | P / R / F1 on known tech skills (500 CVs) | {lx['precision']} / {lx['recall']} / {lx['f1']} | "
            f"{gz['precision']} / {gz['recall']} / {gz['f1']} (gazetteer) |"
        )
    if c3:
        strict = next(r for r in c3["rows"] if r["must_have_coverage"] == 1.0 and r["role_rule"])
        lines.append(
            f"| C3 filter | Good-Fit recall / No-Fit removed / precision of kept (strict) | "
            f"{strict['good_recall']} / {strict['no_fit_removed']} / {strict['precision_of_kept']} | "
            f"1.0 / 0.0 / {c3['baseline_precision']} (no filter) |"
        )
    if c4:
        rows = {r["method"]: r for r in c4["rows"]}
        h, rnd = rows["Hybrid RRF · C1 query (requirements)"], rows["random (baseline)"]
        b = rows["BM25 · C1 query (requirements)"]
        lines.append(
            f"| C4 retrieval | nDCG@10 / MRR / Recall@50 (pooled) | {h['nDCG@10']} / {h['MRR']} / {h['Recall@50']} | "
            f"BM25 {b['nDCG@10']} / {b['MRR']} / {b['Recall@50']}; random {rnd['nDCG@10']} / {rnd['MRR']} / {rnd['Recall@50']} |"
        )
    if c5:
        rows = {r["method"]: r for r in c5["ranking"]}
        rub = rows["Rubric · rules (C5)"]
        hyb = rows["Hybrid RRF score (C4)"]
        exp = next(r for r in c5["ranking"] if r["method"].startswith("PIPELINE"))
        lines += [
            f"| C5 re-ranker | in-pool nDCG@10: rubric only / final blend (see c5 report) | {rub['nDCG@10']} / {exp['nDCG@10']} | "
            f"hybrid {hyb['nDCG@10']}, random {rows['random (baseline)']['nDCG@10']} |",
            f"| C5 re-ranker | macro-F1 3-class (test) | {c5['classification']['rubric_rules_macro_f1']} | "
            f"{c5['classification']['majority_macro_f1']} (majority) |",
        ]
        if c5.get("llm"):
            lines.append(
                f"| C5 LLM scorer | Spearman vs label | {c5['llm']['spearman_llm']} | rules {c5['llm']['spearman_rules_same_pairs']} |"
            )
    if c6:
        s = c6["summary"][0]
        v = {r["injected claim"]: r["detected (recall)"] for r in c6["verifier"]}
        lines += [
            f"| C6 outreach | faithful messages / CTA / length in range | {s['faithful (0 unsupported)']} / {s['CTA present']} / {s['length in range']} | — |",
            f"| C6 verifier | detection recall: fake skill / years / quote / domain | {v.get('fake skill')} / {v.get('wrong years')} / "
            f"{v.get('fake quote')} / {v.get('fake domain')} | FP rate {v.get('none (clean message) → false-positive rate')} |",
            f"| C6 outreach | same-job trigram Jaccard (lower = more personal) | {s['same-job trigram Jaccard (lower=better)']} | — |",
        ]
    if hu.get("top3_precision_good_or_potential") is not None:
        lines += [
            f"| C5 vs manual labels ({hu['fit_labelled']} Djinni pairs, 10 jobs) | top-3 precision (Good or Potential) / share Good | "
            f"{hu['top3_precision_good_or_potential']} / {hu['top3_share_good']} | — |",
            f"| C5 vs manual labels | Spearman final / rubric only; #1 is Good | "
            f"{hu['c5_spearman_final_vs_human']} / {hu['c5_spearman_total_vs_human']}; {hu['top1_good']} | — |",
            f"| C3 vs manual labels | labelled candidates rejected by the system | {hu['labelled_rejected_by_system']} ({hu['labels_of_rejected']}) | — |",
            f"| C6 vs manual labels | mean message score (1–5), messages of the current top 3 | {hu['c6_mean_message_score_top3_sent']} | — |",
        ]
    lines += ["", f"Manual-label agreement: {hu.get('status', 'not run')}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    for key, script in EVALS.items():
        if args.only and key not in args.only:
            continue
        print(f"=== {script}", flush=True)
        r = subprocess.run([sys.executable, script], cwd=HERE, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-3000:])
            raise SystemExit(f"{script} failed")
    text = summary()
    (ROOT / "reports" / "eval_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
