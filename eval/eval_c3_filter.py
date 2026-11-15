"""Eval C3 · Hard filter on fit_labels test (1,759 labelled resume-JD pairs, 71 JDs).

Requirements come from the JD via C1 single-shot extraction, profiles from C2's free-text
path, so this is the filter as it behaves inside the system. A good filter keeps Good/
Potential Fit resumes (recall) and removes No Fit ones (removal rate). Baseline: no filter.
Sweeps the must-have coverage threshold and the role rule.
"""

from __future__ import annotations

from collections import Counter

from common import config, md_table, write_report
from fit_data import bed
from recruiter import c3_filter

COVERAGES = [1.0, 0.75, 0.5, 0.25, 0.0]


def run(b, coverage: float, use_role: bool) -> dict:
    kept = {"Good Fit": 0, "Potential Fit": 0, "No Fit": 0}
    tot = Counter(b.pairs["label"])
    reasons: Counter[str] = Counter()
    for r in b.pairs.itertuples():
        d = c3_filter.check(b.reqs[r.jd_id], b.profiles[r.cv_id], use_role=use_role, min_coverage=coverage)
        if d.passed:
            kept[r.label] += 1
        for reason in d.reasons:
            key = (
                "must-have"
                if reason.startswith("missing")
                else "role"
                if reason.startswith("role")
                else "English"
                if "English" in reason
                else "over-qualified"
                if reason.startswith("over-qualified")
                else "years"
            )
            reasons[key] += 1
    n_kept = sum(kept.values())
    rel_kept = kept["Good Fit"] + kept["Potential Fit"]
    return {
        "must_have_coverage": coverage,
        "role_rule": use_role,
        "kept_share": round(n_kept / len(b.pairs), 4),
        "good_recall": round(kept["Good Fit"] / tot["Good Fit"], 4),
        "good_or_potential_recall": round(rel_kept / (tot["Good Fit"] + tot["Potential Fit"]), 4),
        "no_fit_removed": round(1 - kept["No Fit"] / tot["No Fit"], 4),
        "precision_of_kept": round(rel_kept / n_kept, 4) if n_kept else 0.0,
        "rejection_reasons": dict(reasons),
    }


def main() -> None:
    b = bed("test")
    base_prec = round((b.pairs["gain"] > 0).mean(), 4)
    rows = [run(b, c, role) for role in (False, True) for c in COVERAGES]
    must_counts = Counter(len(r.must_have_skills or []) for r in b.reqs.values())
    yrs = sum(1 for r in b.reqs.values() if r.min_years)
    eng = sum(1 for r in b.reqs.values() if r.english_level not in (None, "no_english"))
    cols = [
        "must_have_coverage",
        "role_rule",
        "kept_share",
        "good_recall",
        "good_or_potential_recall",
        "no_fit_removed",
        "precision_of_kept",
    ]
    md = "\n".join(
        [
            "# C3 · Hard filter: evaluation (fit_labels test)",
            "",
            f"{len(b.pairs)} labelled pairs, {len(b.reqs)} JDs. Requirements extracted by C1 (rules), profiles by C2 (free text).",
            f"Baseline (no filter): keeps 100%, precision of kept = **{base_prec}** (share of Good/Potential pairs).",
            "",
            f"Extracted must-have counts per JD: {dict(sorted(must_counts.items()))}; JDs with a min-years rule: {yrs}; "
            f"with an English rule: {eng}.",
            "",
            md_table(rows, cols),
            "",
            "- `good_recall`: share of Good Fit pairs the filter keeps (losing them is the costly error).",
            "- `no_fit_removed`: share of No Fit pairs removed (the work the filter saves).",
            "- `precision_of_kept`: share of kept pairs that are Good/Potential; compare with the no-filter baseline.",
            "",
            "Rejection reasons at the strict setting (coverage 1.0, role on): "
            + ", ".join(f"{k}: {v}" for k, v in rows[len(COVERAGES)]["rejection_reasons"].items()),
        ]
    )
    write_report(
        "c3_filter",
        {
            "baseline_precision": base_prec,
            "rows": rows,
            "must_have_counts": dict(must_counts),
            "config_default_coverage": config.FILTER_MUST_HAVE_MIN_COVERAGE,
        },
        md,
    )


if __name__ == "__main__":
    main()
