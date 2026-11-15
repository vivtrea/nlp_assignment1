"""Eval C2 · Profile extractor.

A) Skill extraction vs Mehyaar Annotated NER resumes (500 CVs, seed 42), document-level
   micro P/R/F1 on lower-cased skill surface strings.
   Systems: lexicon (ours), gazetteer baseline built from the gold labels of the OTHER
   4,529 CVs (data-driven upper-ish baseline), LLM (if key).
   Reported against ALL gold labels and against gold labels that are known tech skills,
   because the gold contains generic words ("Knowledge", "Building").
B) Free-text field parsing (used for resumes without form fields) vs Djinni form fields
   on 1,000 sampled Djinni CVs: years-of-experience and English level from text.
"""

from __future__ import annotations

import json
import random
import re
import zipfile
from collections import Counter

import pandas as pd

from common import config, md_table, prf, write_report
from recruiter import lexicon, llm
from recruiter.c2_profiles import llm_skills, years_from_text

N_DOCS = 500
LLM_DOCS = 50


def load_ner() -> list[dict]:
    z = zipfile.ZipFile(config.DATA_RAW / "skills_ner" / "ResumesJsonAnnotated.zip")
    docs = []
    for n in sorted(z.namelist()):
        if n.endswith(".json"):
            d = json.loads(z.read(n))
            gold = {a[2].split(":", 1)[1].strip().lower() for a in d.get("annotations", []) if ":" in a[2]}
            docs.append({"name": n, "text": d["text"], "gold": gold})
    return docs


def lexicon_surfaces(text: str) -> set[str]:
    out = set()
    for _canon, pat in lexicon._compiled_skills():
        for m in pat.finditer(text):
            if lexicon._accept(m.group(0)):
                out.add(m.group(0).lower())
    return out


def gazetteer(train_docs: list[dict], min_count: int = 3) -> tuple[re.Pattern[str], set[str]]:
    c = Counter(s for d in train_docs for s in d["gold"])
    terms = sorted((t for t, n in c.items() if n >= min_count and len(t) > 1), key=len, reverse=True)
    alt = "|".join(re.escape(t) for t in terms)
    return re.compile(rf"(?<![\w+#.])(?:{alt})(?![\w+#])", re.I), set(terms)


def gaz_surfaces(text: str, gaz: tuple[re.Pattern[str], set[str]]) -> set[str]:
    # overlapping matches: scan every start position the big alternation can match from
    pat, _terms = gaz
    out, pos = set(), 0
    while True:
        m = pat.search(text, pos)
        if not m:
            return out
        out.add(m.group(0).lower())
        pos = m.start() + 1


def score(docs: list[dict], preds: list[set[str]], known_only: bool) -> dict:
    tp = fp = fn = 0
    for d, p in zip(docs, preds):
        g = {s for s in d["gold"] if lexicon.is_known_skill(s)} if known_only else d["gold"]
        if known_only:
            p = {s for s in p if lexicon.is_known_skill(s)}
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    return prf(tp, fp, fn)


def eval_ner() -> dict:
    docs = load_ner()
    rnd = random.Random(config.SEED)
    idx = set(rnd.sample(range(len(docs)), N_DOCS))
    test = [d for i, d in enumerate(docs) if i in idx]
    train = [d for i, d in enumerate(docs) if i not in idx]
    pats = gazetteer(train)

    systems = {
        "lexicon (ours)": [lexicon_surfaces(d["text"]) for d in test],
        "gazetteer from other 4,529 gold CVs (baseline)": [gaz_surfaces(d["text"], pats) for d in test],
    }
    llm_rows = []
    if config.llm_enabled():
        sub = test[:LLM_DOCS]
        llm.reset_stats()
        preds = [{s.lower() for s in llm_skills(d["text"])} for d in sub]
        fallback = llm.STATS["fallbacks"] / max(1, llm.STATS["fallbacks"] + llm.STATS["llm_ok"])
        llm_rows = [
            {"system": f"LLM ({LLM_DOCS} docs, fallback rate {fallback:.0%})", "gold": g, **score(sub, preds, g == "known")}
            for g in ("all", "known")
        ]
        llm_rows += [
            {
                "system": f"lexicon on same {LLM_DOCS} docs",
                "gold": g,
                **score(sub, [lexicon_surfaces(d["text"]) for d in sub], g == "known"),
            }
            for g in ("all", "known")
        ]

    rows = []
    for name, preds in systems.items():
        for g in ("all", "known"):
            rows.append({"system": name, "gold": g, **score(test, preds, g == "known")})
    rows += llm_rows

    all_gold = Counter(s for d in test for s in d["gold"])
    n_ann = sum(all_gold.values())
    known_share = sum(n for s, n in all_gold.items() if lexicon.is_known_skill(s)) / n_ann
    missed = Counter(s for d, p in zip(test, systems["lexicon (ours)"]) for s in d["gold"] - p)
    fps = Counter(s for d, p in zip(test, systems["lexicon (ours)"]) for s in p - d["gold"])
    return {
        "n_docs": len(test),
        "gold_unique_per_doc_mean": round(n_ann / len(test), 2),
        "gold_share_known_tech_skills": round(known_share, 4),
        "rows": rows,
        "top_missed_gold": missed.most_common(25),
        "top_false_positives": fps.most_common(15),
    }


def eval_fields(n: int = 1000) -> dict:
    df = pd.read_parquet(config.DATA_RAW / "djinni_cvs" / "train.parquet").sample(n, random_state=config.SEED)
    y_cov = y_close = 0
    y_err = []
    e_cov = e_ok = e_n = 0
    for r in df.to_dict("records"):
        text = f"{r['Position']}\n{r['CV']}"
        y = years_from_text(text)
        if y is not None:
            y_cov += 1
            gold = min(float(r["Experience Years"]), 11.0)
            y_err.append(abs(min(y, 11.0) - gold))
            y_close += abs(min(y, 11.0) - gold) <= 1
        if r["English Level"]:
            e_n += 1
            e = lexicon.detect_english(text)
            if e:
                e_cov += 1
                e_ok += e == r["English Level"]
    return {
        "n": n,
        "years_coverage": round(y_cov / n, 4),
        "years_mae_when_found": round(sum(y_err) / len(y_err), 3) if y_err else None,
        "years_within_1_when_found": round(y_close / y_cov, 4) if y_cov else None,
        "english_coverage": round(e_cov / e_n, 4),
        "english_acc_when_found": round(e_ok / e_cov, 4) if e_cov else None,
    }


def main() -> None:
    ner = eval_ner()
    fields = eval_fields()
    md = "\n".join(
        [
            "# C2 · Profile extractor: evaluation",
            "",
            "## A) Skill extraction vs. Mehyaar annotated resumes",
            "",
            f"{ner['n_docs']} CVs (seed 42). Document-level micro P/R/F1 on lower-cased surface strings. "
            f"Gold has {ner['gold_unique_per_doc_mean']} unique skill strings per CV; only "
            f"**{ner['gold_share_known_tech_skills']:.1%}** of gold annotations are known tech skills (our lexicon). "
            "`gold=known` restricts both gold and predictions to known skills (measures recall of real tech skills).",
            "",
            md_table(ner["rows"], ["system", "gold", "precision", "recall", "f1", "tp", "fp", "fn"]),
            "",
            "### Most frequent gold strings the lexicon misses",
            "",
            ", ".join(f"`{s}` ({n})" for s, n in ner["top_missed_gold"]),
            "",
            "### Most frequent lexicon predictions not in gold",
            "",
            ", ".join(f"`{s}` ({n})" for s, n in ner["top_false_positives"]),
            "",
            "## B) Parsing years / English from free text vs. Djinni form fields (1,000 CVs)",
            "",
            md_table([fields]),
            "",
            "Used only for resumes without structured fields (fit_labels); Djinni profiles use the form fields directly.",
        ]
    )
    write_report("c2_profiles", {"ner": ner, "fields_from_text": fields}, md)


if __name__ == "__main__":
    main()
