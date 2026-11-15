"""Eval C6 · Outreach writer + claim verifier.

20 Djinni job descriptions (seed 42) -> C1 single-shot -> pipeline (rule scorer, so every writer
sees the same shortlist) -> top-3 messages per job.
A) Message quality (automatic, per message): faithfulness (0 unsupported claims), claims
   checked, personalisation (# verified candidate-specific facts), length within
   [OUTREACH_MIN_CHARS, OUTREACH_MAX_CHARS], CTA present, cross-candidate similarity
   (mean pairwise word-trigram Jaccard of messages for the same job; lower = more personal).
B) Verifier quality: inject one hallucination of each type into every message and measure
   detection recall per type; false-positive rate = clean messages flagged.
C) LLM-as-judge with eval/rubrics/outreach.md when an API key is set.
"""

from __future__ import annotations

import random
import re
from itertools import combinations

import pandas as pd
from pydantic import BaseModel

from common import config, md_table, mean, write_report
from recruiter import lexicon, llm, pipeline
from recruiter.c1_requirements import requirements_from_jd
from recruiter.c5_reranker import RuleScorer
from recruiter.c6_outreach import LLMWriter, TemplateWriter, verify_claims

N_JOBS = 20
CTA_RE = re.compile(r"\b(?:call|chat|talk|connect|meet)\b[^.?!]*\?", re.I)


def trigrams(text: str) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return set(zip(w, w[1:], w[2:]))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def personal_facts(msg: str, p) -> int:
    n = sum(1 for s in set(lexicon.extract_skills(msg)) if s in p.skills)
    n += sum(1 for q in re.findall(r"[\"“]([^\"”]{8,})[\"”]", msg) if q.lower().strip(" .") in p.evidence_text().lower())
    n += bool(p.title and p.title.lower() in msg.lower())
    return n


def inject(msg: str, p, rnd: random.Random) -> dict[str, str]:
    absent = [s for s in lexicon.SKILLS if s not in p.skills and s.lower() not in p.evidence_text().lower()]
    dom = [d for d in lexicon.DOMAINS if d not in p.domains]
    y = (p.years_experience or 0) + 4
    out = {
        "fake skill": f"Your hands-on experience with {rnd.choice(absent)} is impressive.",
        "wrong years": f"You have {y:g} years of experience in the field.",
        "fake quote": 'Your note "led a team of 40 engineers at a unicorn startup" stood out.',
        "fake domain": f"Your background in {lexicon.DOMAINS[rnd.choice(dom)][0]} projects is great.",
    }
    head, _, tail = msg.partition("\n\n")
    return {k: f"{head}\n{v}\n\n{tail}" for k, v in out.items()}


class _Judge(BaseModel):
    personalisation: int
    relevance: int
    tone: int
    call_to_action: int
    comment: str = ""


def main() -> None:
    jobs = pd.read_parquet(config.DATA_RAW / "djinni_jobs" / "train.parquet").sample(N_JOBS, random_state=config.SEED)
    writers = [("template (baseline)", TemplateWriter())]
    if config.llm_enabled():
        writers.append(("llm + verifier", LLMWriter()))
    rnd = random.Random(config.SEED)
    summary, verifier_rows, examples, judge_rows = [], [], [], []
    for wname, writer in writers:
        msgs = []
        for r in jobs.to_dict("records"):
            req = requirements_from_jd(f"{r['Position']}\n{r['Long Description']}")
            req.title = r["Position"]
            res = pipeline.run(req, writer=writer, scorer=RuleScorer())  # same shortlist for every writer
            for m in res.messages:
                msgs.append((r["id"], req, res.profiles[m.candidate_id], m))
        if not examples and msgs:
            examples = [m.text for *_, m in msgs[:2]]
        per_job: dict[str, list[str]] = {}
        for jid, _, _, m in msgs:
            per_job.setdefault(jid, []).append(m.text)
        sims = [jaccard(trigrams(a), trigrams(b)) for ts in per_job.values() for a, b in combinations(ts, 2)]
        summary.append(
            {
                "writer": wname,
                "messages": len(msgs),
                "faithful (0 unsupported)": mean([m.faithful for *_, m in msgs]),
                "claims checked / msg": mean([m.claims_checked for *_, m in msgs]),
                "verified personal facts / msg": mean([personal_facts(m.text, p) for _, _, p, m in msgs]),
                "length in range": mean(
                    [config.OUTREACH_MIN_CHARS <= len(m.text) <= config.OUTREACH_MAX_CHARS for *_, m in msgs]
                ),
                "mean chars": mean([len(m.text) for *_, m in msgs]),
                "CTA present": mean([bool(CTA_RE.search(m.text)) for *_, m in msgs]),
                "same-job trigram Jaccard (lower=better)": mean(sims),
                "fell back to template": mean([m.writer == "template" for *_, m in msgs])
                if wname.startswith("llm")
                else float("nan"),
            }
        )
        if wname.startswith("template"):
            det: dict[str, list[bool]] = {}
            fp = []
            for _, req, p, m in msgs:
                fp.append(bool(verify_claims(m.text, p, req)[1]))
                for kind, bad in inject(m.text, p, rnd).items():
                    det.setdefault(kind, []).append(bool(verify_claims(bad, p, req)[1]))
            verifier_rows = [{"injected claim": k, "n": len(v), "detected (recall)": mean(v)} for k, v in det.items()]
            verifier_rows.append(
                {"injected claim": "none (clean message) → false-positive rate", "n": len(fp), "detected (recall)": mean(fp)}
            )
        if config.llm_enabled():
            rub = (config.ROOT / "eval" / "rubrics" / "outreach.md").read_text()
            scores, judge_failed = [], 0
            for _, req, p, m in msgs[:30]:
                prompt = (
                    f"{rub}\n\nJob: {req.model_dump_json(exclude={'summary'})}\nCandidate CV: {p.cv_text[:2500]}\n"
                    f"Looking for: {p.looking_for}\n\nMessage:\n{m.text}\n\nReturn ONLY JSON with keys personalisation, "
                    "relevance, tone, call_to_action (1-5 each) and comment."
                )
                try:
                    scores.append(llm.complete_json(prompt, _Judge, component="C6-judge", model=config.JUDGE_MODEL))
                except (llm.LLMUnavailable, ValueError):
                    judge_failed += 1
            judge_rows.append(
                {
                    "writer": wname,
                    "n": len(scores),
                    "judge calls failed": judge_failed,
                    **{
                        k: mean([getattr(s, k) for s in scores])
                        for k in ("personalisation", "relevance", "tone", "call_to_action")
                    },
                }
            )

    md = "\n".join(
        [
            "# C6 · Outreach writer: evaluation",
            "",
            f"{N_JOBS} Djinni JDs (seed 42) → C1 → C3 → C4 → C5 → top-{config.OUTREACH_TOP_K} messages per job.",
            "",
            "## A) Message quality",
            "",
            md_table(summary),
            "",
            "## B) Claim verifier: hallucination injection",
            "",
            "One fabricated sentence of each type is inserted into each template message; the verifier should flag it.",
            "",
            md_table(verifier_rows),
            "",
            "## C) LLM-as-judge (rubric: eval/rubrics/outreach.md)",
            "",
            md_table(judge_rows) if judge_rows else "_Not run: no OPENROUTER_API_KEY._",
            "",
            "## Example messages",
            "",
            *[f"```\n{e}\n```\n" for e in examples],
        ]
    )
    write_report("c6_outreach", {"summary": summary, "verifier": verifier_rows, "judge": judge_rows, "examples": examples}, md)


if __name__ == "__main__":
    main()
