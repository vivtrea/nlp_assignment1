"""C5 · Rubric re-ranker: JobRequirements + CandidateProfile -> scored, evidenced ranking.

Rubric adapted from interview evaluation forms (klearskill.com): six criteria scored 1-5,
weights in config.RUBRIC_WEIGHTS. Every score needs an evidence quote from the candidate
record; a criterion without a verifiable quote scores 1 (CLAUDE.md §5).

Two scorers:
- `RuleScorer` (baseline, deterministic, offline)
- `LLMScorer` (LLM-as-scorer, temperature 0, JSON output, evidence verified as substring)
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ValidationError

from recruiter import config, lexicon, llm
from recruiter.c3_filter import has_skill
from recruiter.schemas import CandidateProfile, CriterionScore, JobRequirements, RankedCandidate

CRITERIA = list(config.RUBRIC_WEIGHTS)
NEGATION = re.compile(r"\b(?:not|don'?t|do not|no|never|avoid|except|without)\b[^.]{0,40}$", re.I)
PROTECTED = re.compile(
    r"\b(?:he|she|his|her|him|hers|male|female|married|single|\d{1,2}\s*years?\s*old|age[d]?\s*\d{2}|nationality|citizen(?:ship)?)\b",
    re.I,
)


_BOUNDARY = re.compile(r"[.!?](?=\s|$)|\n")


def sentence_with(text: str, span: tuple[int, int], max_len: int = 220) -> str:
    """Sentence around `span`. A '.' ends a sentence only before whitespace, so '4.5 years',
    'Node.js' and 'ASP.NET' are never cut."""
    before = [m.end() for m in _BOUNDARY.finditer(text, 0, span[0])]
    start = before[-1] if before else 0
    m_end = _BOUNDARY.search(text, span[1])
    end = m_end.start() if m_end else len(text)
    quote = text[start:end].strip()
    if len(quote) > max_len:
        a = max(0, span[0] - start - max_len // 2)
        b = a + max_len
        # snap to word boundaries so a quote never starts or ends mid-word
        while a > 0 and quote[a - 1].isalnum():
            a += 1
        while b < len(quote) and quote[b - 1].isalnum() and quote[b].isalnum():
            b -= 1
        quote = quote[a:b].strip()
    return quote


def quote_skill(p: CandidateProfile, skill: str) -> str:
    text = p.evidence_text()
    span = lexicon.find_skill_span(text, skill)
    return sentence_with(text, span) if span else ""


def total_score(criteria: list[CriterionScore]) -> float:
    by = {c.criterion: c.score for c in criteria}
    return round(sum(w * (by.get(k, 1) - 1) / 4 for k, w in config.RUBRIC_WEIGHTS.items()), 4)


def _eng(level: str | None) -> int | None:
    return config.ENGLISH_ORDER.index(level) if level in config.ENGLISH_ORDER else None


class RuleScorer:
    name = "rules"

    def score(self, req: JobRequirements, p: CandidateProfile) -> list[CriterionScore]:
        return [
            self._must(req, p),
            self._experience(req, p),
            self._seniority(req, p),
            self._nice(req, p),
            self._english(req, p),
            self._motivation(req, p),
        ]

    def _must(self, req: JobRequirements, p: CandidateProfile) -> CriterionScore:
        must = req.must_have_skills or []
        if not must:
            return CriterionScore(criterion="must_have_skills", score=3, rationale="no must-haves specified")
        quotes = {s: quote_skill(p, s) for s in must if has_skill(p, s)}
        frac = len(quotes) / len(must)
        score = 1 + round(4 * frac) if quotes else 1
        ev = " | ".join(q for q in quotes.values() if q)[:400]
        return CriterionScore(
            criterion="must_have_skills",
            score=score if ev else 1,
            evidence=ev,
            rationale=f"{len(quotes)}/{len(must)} must-haves evidenced",
        )

    def _experience(self, req: JobRequirements, p: CandidateProfile) -> CriterionScore:
        if p.years_experience is None:
            return CriterionScore(criterion="relevant_experience", score=1, rationale="years unknown")
        diff = p.years_experience - (req.min_years or 0)
        score = 5 if diff >= 3 else 4 if diff >= 1 else 3 if diff >= 0 else 2 if diff >= -1 else 1
        role_ok = req.role is None or p.role is None or p.role in lexicon.related_roles(req.role)
        if not role_ok:
            score = max(1, score - 2)
        # over-qualification: a lead is not a better trainee. If the job states a level and the
        # candidate is >= 2 levels above it, extra years stop counting.
        if req.seniority in config.SENIORITY_ORDER and p.seniority in config.SENIORITY_ORDER:
            gap = config.SENIORITY_ORDER.index(p.seniority) - config.SENIORITY_ORDER.index(req.seniority)
            if gap >= 2:
                score = min(score, 3)
        years_txt = "10+" if p.years_experience >= 11 else f"{p.years_experience:g}"
        ev = f"[profile] {p.title} · experience: {years_txt} years"
        return CriterionScore(
            criterion="relevant_experience",
            score=score,
            evidence=ev,
            rationale=f"{years_txt}y vs min {req.min_years or 0:g}y; role match={role_ok}",
        )

    def _seniority(self, req: JobRequirements, p: CandidateProfile) -> CriterionScore:
        need = req.seniority
        if need == "unknown" and req.min_years is not None:
            need = "junior" if req.min_years < 2 else "middle" if req.min_years < 5 else "senior"
        if need == "unknown" or p.seniority == "unknown":
            return CriterionScore(
                criterion="seniority_fit",
                score=3 if need == "unknown" else 1,
                rationale="seniority not specified" if need == "unknown" else "candidate seniority unknown",
            )
        d = abs(config.SENIORITY_ORDER.index(need) - config.SENIORITY_ORDER.index(p.seniority))
        score = {0: 5, 1: 3, 2: 2}.get(d, 1)
        return CriterionScore(
            criterion="seniority_fit",
            score=score,
            evidence=f"[profile] {p.title} ({p.seniority})",
            rationale=f"needs {need}, has {p.seniority}",
        )

    def _nice(self, req: JobRequirements, p: CandidateProfile) -> CriterionScore:
        items = list(req.nice_to_have_skills)
        if not items and not req.domain:
            return CriterionScore(criterion="nice_to_have", score=3, rationale="no nice-to-haves specified")
        quotes = [quote_skill(p, s) for s in items if has_skill(p, s)]
        hits = len(quotes)
        total = len(items)
        if req.domain:
            total += 1
            if req.domain in p.domains:
                hits += 1
                kw = next((k for k in lexicon.DOMAINS[req.domain] if k in p.evidence_text().lower()), req.domain)
                span = lexicon.find_skill_span(p.evidence_text(), kw)
                if span:
                    quotes.append(sentence_with(p.evidence_text(), span))
        ev = " | ".join(q for q in quotes if q)[:400]
        score = 1 + round(4 * hits / total) if ev else 1
        return CriterionScore(criterion="nice_to_have", score=score, evidence=ev, rationale=f"{hits}/{total} nice-to-haves")

    def _english(self, req: JobRequirements, p: CandidateProfile) -> CriterionScore:
        have, need = _eng(p.english_level), _eng(req.english_level) or 0
        if have is None:
            return CriterionScore(criterion="english", score=1, rationale="English level unknown")
        if need == 0:
            score = {5: 5, 4: 4, 3: 3}.get(have, 2)
        else:
            score = 5 if have > need else 4 if have == need else 2 if have == need - 1 else 1
        return CriterionScore(
            criterion="english",
            score=score,
            evidence=f"[profile] English: {p.english_level}",
            rationale=f"has {p.english_level}, needs {req.english_level}",
        )

    def _motivation(self, req: JobRequirements, p: CandidateProfile) -> CriterionScore:
        lf = p.looking_for
        if not lf:
            return CriterionScore(criterion="motivation", score=1, rationale="no 'Looking For' statement")
        targets = [s for s in (req.must_have_skills or []) + req.nice_to_have_skills]
        if req.role:
            targets.append(req.role)
        hits, quotes = 0, []
        for t in targets:
            span = lexicon.find_skill_span(lf, t)
            if span:
                hits += 1
                quotes.append(sentence_with(lf, span))
        if req.domain:
            for kw in lexicon.DOMAINS.get(req.domain, []):
                m = re.search(rf"\b{re.escape(kw)}\b", lf, re.I)
                if m:
                    q = sentence_with(lf, (m.start(), m.end()))
                    if NEGATION.search(lf[: m.start()]):
                        return CriterionScore(
                            criterion="motivation", score=1, evidence=q, rationale=f"explicitly avoids {req.domain}"
                        )
                    hits += 1
                    quotes.append(q)
                    break
        if not quotes:
            return CriterionScore(
                criterion="motivation", score=2, evidence=lf[:200], rationale="states preferences, none overlap with the job"
            )
        return CriterionScore(
            criterion="motivation",
            score=min(5, 2 + hits),
            evidence=" | ".join(dict.fromkeys(quotes))[:400],
            rationale=f"{hits} job aspects match 'Looking For'",
        )


class _LLMOut(BaseModel):
    criteria: list[CriterionScore]


def redact(text: str) -> str:
    return PROTECTED.sub("[redacted]", text)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def verify_evidence(c: CriterionScore, p: CandidateProfile) -> CriterionScore:
    """Grounding check: the quote must literally appear in the record, else score 1."""
    ev = c.evidence.strip()
    if ev.startswith("[profile]"):
        return c
    parts = [x for x in (q.strip().strip('"') for q in ev.split("|")) if x]
    hay = _norm(redact(p.evidence_text()))
    if not parts or not all(_norm(x) in hay for x in parts):
        return c.model_copy(update={"score": 1, "rationale": f"evidence not found in CV; {c.rationale}"[:300]})
    return c


class LLMScorer:
    name = "llm"

    def __init__(self) -> None:
        self.fallback = RuleScorer()

    def score(self, req: JobRequirements, p: CandidateProfile) -> list[CriterionScore]:
        if not config.llm_enabled():
            return self.fallback.score(req, p)
        rules = {c.criterion: c for c in self.fallback.score(req, p)}
        prompt = llm.load_prompt(
            "c5_rubric",
            requirements=req.model_dump_json(exclude={"summary"}),
            job_summary=req.summary[:1200],
            title=p.title,
            years="unknown" if p.years_experience is None else f"{p.years_experience:g}",
            english=p.english_level,
            looking_for=redact(p.looking_for) or "(none)",
            cv=redact(p.cv_text)[:5000],
        )
        try:
            out = llm.complete_json(prompt, _LLMOut, component="C5", temperature=config.TEMPERATURE_SCORING)
        except (llm.LLMUnavailable, ValidationError, ValueError):
            llm.note(False)
            return list(rules.values())
        llm.note(True)
        got = {c.criterion: verify_evidence(c, p) for c in out.criteria if c.criterion in CRITERIA}
        # structured-field criteria stay deterministic; missing criteria fall back to rules
        for k in ("english",):
            got[k] = rules[k]
        return [got.get(k, rules[k]) for k in CRITERIA]


def rerank(
    req: JobRequirements,
    profiles: list[CandidateProfile],
    scorer: RuleScorer | LLMScorer | None = None,
    retrieval: dict[str, tuple[int, float]] | None = None,
    alpha: float = config.RERANK_BLEND_ALPHA,
    llm_top: int = config.LLM_RERANK_TOP,
) -> list[RankedCandidate]:
    """Two-stage re-rank.

    Stage 1: the rule scorer scores every candidate; blend with retrieval and sort.
    Stage 2 (LLM mode): the LLM re-scores only the first `llm_top`; those are re-blended
    among themselves and stay ahead of the rest, so the LLM refines the order of the
    shortlist but never pulls in a candidate that stage 1 ranked below it.
    `scorer=RuleScorer()` forces rules only; `scorer=LLMScorer()` or None (with a key) enables stage 2.
    """
    use_llm = config.llm_enabled() and (isinstance(scorer, LLMScorer) or scorer is None)
    rules = RuleScorer()
    stage1 = [_ranked(p, rules.score(req, p), retrieval, "rules") for p in profiles]
    stage1 = blend_and_sort(stage1, alpha)
    if not use_llm or not stage1:
        return stage1
    llm_scorer = scorer if isinstance(scorer, LLMScorer) else LLMScorer()
    by_id = {p.candidate_id: p for p in profiles}
    head = []
    for c in stage1[:llm_top]:
        before = llm.STATS["fallbacks"]
        crit = llm_scorer.score(req, by_id[c.candidate_id])
        head.append(_ranked(by_id[c.candidate_id], crit, retrieval, "rules" if llm.STATS["fallbacks"] > before else "llm"))
    return blend_and_sort(head, alpha) + stage1[llm_top:]


def _ranked(
    p: CandidateProfile, crit: list[CriterionScore], retrieval: dict[str, tuple[int, float]] | None, scorer: str
) -> RankedCandidate:
    r = (retrieval or {}).get(p.candidate_id, (None, None))
    return RankedCandidate(
        candidate_id=p.candidate_id,
        total=total_score(crit),
        criteria=crit,
        retrieval_rank=r[0],
        retrieval_score=r[1],
        scorer=scorer,
    )


def _minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 0.5 for x in xs]


def blend_and_sort(cands: list[RankedCandidate], alpha: float = config.RERANK_BLEND_ALPHA) -> list[RankedCandidate]:
    """final = alpha * rubric + (1 - alpha) * retrieval, min-max normalised within this query."""
    if not cands:
        return cands
    rub = _minmax([c.total for c in cands])
    has_retrieval = all(c.retrieval_score is not None for c in cands)
    ret = _minmax([c.retrieval_score for c in cands]) if has_retrieval else rub
    for c, r, h in zip(cands, rub, ret):
        c.final = round(alpha * r + (1 - alpha) * h, 4) if has_retrieval else round(r, 4)
    return sorted(cands, key=lambda c: (-c.final, -c.total, c.retrieval_rank if c.retrieval_rank is not None else 1e9))
