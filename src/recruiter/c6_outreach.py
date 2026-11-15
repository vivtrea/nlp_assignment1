"""C6 · Outreach writer + claim verifier.

Writers: `TemplateWriter` (baseline, grounded by construction) and `LLMWriter`
(temperature <= 0.7, grounded prompt). Every message goes through `verify_claims`:
claims about the candidate (sentences addressed to "you/your") must be supported by the
candidate record; job claims must be supported by the requirements. An LLM message
with unsupported claims is regenerated once with feedback, then replaced by the template.
"""

from __future__ import annotations

import re

from recruiter import config, lexicon, llm
from recruiter.schemas import CandidateProfile, JobRequirements, OutreachMessage, RankedCandidate

YOU_RE = re.compile(r"\b(?:you|your|you've|you're|yours)\b", re.I)
WE_RE = re.compile(r"\b(?:we|our|we're|the role|this role|the position|the team)\b", re.I)
QUOTE_RE = re.compile(r"[\"“]([^\"”]{8,})[\"”]")
YEARS_RE = re.compile(r"(\d+(?:\.\d)?)\+?\s*(?:years?|yrs?)", re.I)


RESTRICTION = re.compile(r"\b(?:no|not|don'?t|do not|never|without|except|only|avoid|won'?t|wont|isn'?t|aren'?t)\b", re.I)


GOAL = re.compile(
    r"look|want|wish|interest|seek|grow|develop|opportunit|prefer|ideal|would like|career|project|team|role|"
    r"position|product|compan|learn|challeng|goal|expect|focus|environment|culture|long[- ]term",
    re.I,
)


def _usable_quote(s: str) -> bool:
    """Not a list header ('My achievements include the following:') and a real sentence."""
    return not s.rstrip().endswith(":") and len(s.split()) >= 4


def _goal_sentence(looking_for: str) -> str:
    """First 'Looking For' sentence that states a goal. Restrictions ('No gambling',
    'GMT+3 only', 'not Angular.js'), list headers and remarks without a goal word
    ('mobile manual testing is pretty routine') are skipped."""
    for s in _sentences(looking_for):
        s = s.lstrip("-•* ").strip()
        if len(s) >= 15 and _usable_quote(s) and GOAL.search(s) and not RESTRICTION.search(s):
            return s
    return ""


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def verify_claims(msg: str, p: CandidateProfile, req: JobRequirements) -> tuple[int, list[str]]:
    """Return (#claims checked, unsupported claims)."""
    record = _norm(p.evidence_text())
    cand_skills = set(p.skills) | {s for s in lexicon.extract_skills(p.evidence_text())}
    job_skills = set(req.must_have_skills or []) | set(req.nice_to_have_skills)
    job_text = _norm(req.summary + " " + (req.title or "") + " " + (req.role or ""))
    checked, bad = 0, []
    for q in QUOTE_RE.findall(msg):
        checked += 1
        if _norm(q.strip(" .…")) not in record:
            bad.append(f'quote not in CV: "{q[:60]}"')
    for s in _sentences(msg):
        about_you, about_job = bool(YOU_RE.search(s)), bool(WE_RE.search(s))
        skills = lexicon.extract_skills(s)
        if about_you and not about_job:
            for sk in skills:
                checked += 1
                if sk not in cand_skills:
                    bad.append(f"candidate skill not in CV: {sk}")
            for y in YEARS_RE.findall(s):
                checked += 1
                yv = float(y)
                have = p.years_experience
                if have is None or not (abs(yv - have) < 0.6 or (have >= 11 and yv >= 10)):
                    bad.append(f"years claim {y} vs profile {have}")
            for d in lexicon.detect_domains(s):
                checked += 1
                if d not in p.domains:
                    bad.append(f"candidate domain not in CV: {d}")
        elif about_job:
            for sk in skills:
                checked += 1
                if sk not in job_skills and _norm(sk) not in job_text and sk not in cand_skills:
                    bad.append(f"job skill not in requirements: {sk}")
    return checked, bad


def _years_phrase(y: float | None) -> str:
    if y is None:
        return ""
    if y >= 11:
        return "10+ years"
    if y < 1:
        return ""
    return f"{y:g} years" if y != 1 else "1 year"


def _short_quote(ranked: RankedCandidate | None, p: CandidateProfile) -> str:
    """A short CV quote, preferring must-have evidence from C5."""
    if ranked:
        for c in ranked.criteria:
            if c.criterion in ("must_have_skills", "nice_to_have") and c.evidence and not c.evidence.startswith("[profile]"):
                q = c.evidence.split("|")[0].strip()
                if 15 <= len(q) <= 160 and _usable_quote(q) and _norm(q) in _norm(p.cv_text) and _norm(q) != _norm(p.title):
                    return q
    for s in _sentences(p.cv_text):
        if 30 <= len(s) <= 160 and _usable_quote(s):
            return s
    return ""


class TemplateWriter:
    name = "template"

    def write(self, req: JobRequirements, p: CandidateProfile, ranked: RankedCandidate | None = None) -> str:
        role_name = (
            req.title
            or f"{req.seniority.capitalize() + ' ' if req.seniority != 'unknown' else ''}{req.role or 'engineering'} role"
        )
        matched = [s for s in (req.must_have_skills or []) + req.nice_to_have_skills if s in p.skills][:3]
        yrs = _years_phrase(p.years_experience)
        lines = ["Hi there,", ""]
        intro = f"I came across your profile as {p.title}" if p.title else "I came across your profile"
        if yrs:
            intro += f" with {yrs} of experience"
        lines.append(intro + ".")
        if matched:
            lines.append(f"Your background in {', '.join(matched)} is exactly what caught my attention.")
        q = _short_quote(ranked, p)
        if q:
            lines.append(f'In particular, this line stood out: "{q.lstrip("-•* ").rstrip(".")}".')
        lines += ["", f"We are hiring for a {role_name}."]
        if req.must_have_skills:
            lines.append(f"The role needs strong {', '.join(req.must_have_skills[:4])}.")
        lf = _goal_sentence(p.looking_for)
        if len(lf) > 140:
            lf = lf[:140].rsplit(" ", 1)[0].rstrip(",;:") + "…"
        if lf:
            lines.append(f'You mentioned you are looking for "{lf.rstrip(".")}", so I think this could be a good match.')
        lines += ["", "Would you be open to a short 15-minute call this week to talk it through?", "", "Best regards"]
        return "\n".join(lines)


class LLMWriter:
    name = "llm"

    def __init__(self) -> None:
        self.fallback = TemplateWriter()

    def write(self, req: JobRequirements, p: CandidateProfile, ranked: RankedCandidate | None = None, feedback: str = "") -> str:
        if not config.llm_enabled():
            return self.fallback.write(req, p, ranked)
        evidence = "\n".join(f"- {c.criterion}: {c.evidence}" for c in (ranked.criteria if ranked else []) if c.evidence)
        prompt = llm.load_prompt(
            "c6_outreach",
            requirements=req.model_dump_json(exclude={"summary"}),
            title=p.title,
            years=_years_phrase(p.years_experience) or "unknown",
            skills=", ".join(p.skills[:25]),
            looking_for=p.looking_for or "(none)",
            evidence=evidence or "(none)",
            cv=p.cv_text[:3000],
            min_chars=config.OUTREACH_MIN_CHARS,
            max_chars=config.OUTREACH_MAX_CHARS,
            feedback=feedback or "(none)",
        )
        return llm.complete(prompt, component="C6", temperature=config.TEMPERATURE_WRITING).strip()


def write_message(
    req: JobRequirements,
    p: CandidateProfile,
    ranked: RankedCandidate | None = None,
    writer: TemplateWriter | LLMWriter | None = None,
) -> OutreachMessage:
    writer = writer or (LLMWriter() if config.llm_enabled() else TemplateWriter())
    used = "llm" if isinstance(writer, LLMWriter) and config.llm_enabled() else "template"
    try:
        text = writer.write(req, p, ranked)
        if used == "llm":
            llm.note(True)
    except llm.LLMUnavailable:  # rate limit / outage after retries: degrade to the grounded template
        llm.note(False)
        writer, used = TemplateWriter(), "template"
        text = writer.write(req, p, ranked)
    n, bad = verify_claims(text, p, req)
    if bad and used == "llm":
        try:
            text = writer.write(req, p, ranked, feedback="Remove these unsupported claims: " + "; ".join(bad))
            n, bad = verify_claims(text, p, req)
        except llm.LLMUnavailable:
            llm.note(False)
        if bad:
            text = TemplateWriter().write(req, p, ranked)
            n, bad = verify_claims(text, p, req)
            used = "template"
    return OutreachMessage(candidate_id=p.candidate_id, text=text, writer=used, claims_checked=n, unsupported_claims=bad)
