"""C1 · Requirement collector: chat turns -> JobRequirements (slot-filling loop).

Two extractors share one dialogue policy:
- `RuleExtractor` (baseline, offline): lexicon + regex.
- `LLMExtractor`: JSON-schema output validated with pydantic.
The policy asks for the first missing REQUIRED slot until all are filled.
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, ValidationError

from recruiter import config, lexicon, llm
from recruiter.schemas import JobRequirements

QUESTIONS = {
    "role": "What role are you hiring for (e.g. Python backend developer, QA automation, product designer)?",
    "must_have_skills": "Which skills are must-haves? List the non-negotiable technologies or competencies.",
    "min_years": "How many years of relevant experience should candidates have at minimum?",
    "english_level": "What English level is required (basic, pre-intermediate, intermediate, upper-intermediate, fluent, or no requirement)?",
}
DONE_MESSAGE = "Thanks, I have everything I need. Searching the candidate database now."

NONE_ANSWER = re.compile(
    r"^\s*(?:no|none|any|not (?:required|important|needed)|doesn'?t matter|no requirement|n/?a|nothing)\b", re.I
)
NICE_CUES = re.compile(
    r"nice to have|would be a plus|is a plus|are a plus|as a plus|bonus|preferred|optional|advantage|desirable|good to have|ideally",
    re.I,
)
MUST_CUES = re.compile(
    r"\b(?:must|required|requirements?|mandatory|should (?:know|have)|experience (?:with|in)|knowledge of|skills?|stack|proficien\w*)\b",
    re.I,
)
YEARS_RE = re.compile(r"(\d+(?:[.,]\d)?)\s*\+?\s*(?:-|to|–)?\s*(?:\d+\s*)?(?:years?|yrs?|y\.?o\.?e|рок)", re.I)
WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "ten": 10}


class Extractor(Protocol):
    def extract(self, text: str, current: JobRequirements, asked: str | None) -> JobRequirements: ...


def _years(text: str, asked: str | None) -> float | None:
    m = YEARS_RE.search(text)
    if m:
        return float(m.group(1).replace(",", "."))
    if re.search(r"\b(?:no experience|no commercial experience|entry[- ]level|without experience|no_exp)\b", text, re.I):
        return 0.0
    for w, n in WORD_NUM.items():
        if re.search(rf"\b{w}\+?\s+years?\b", text, re.I):
            return float(n)
    if asked == "min_years":
        m = re.search(r"\b(\d+(?:\.\d)?)\b", text)
        if m:
            return float(m.group(1))
    return None


def split_must_nice(text: str) -> tuple[list[str], list[str]]:
    """Skills in sentences/clauses with 'nice to have' cues are nice-to-have."""
    must: list[str] = []
    nice: list[str] = []
    in_nice_section = False
    for chunk in re.split(r"(?<=[.!?;\n])\s+|\n", text):
        if not chunk.strip():
            continue
        header_like = len(chunk) < 60 and chunk.strip().endswith(":")
        if header_like:
            in_nice_section = bool(NICE_CUES.search(chunk))
        if in_nice_section or not NICE_CUES.search(chunk):
            for s in lexicon.extract_skills(chunk):
                (nice if in_nice_section else must).append(s)
            continue
        # mixed sentence ("React, Redux required, Next.js is a plus"): decide per clause
        for clause in re.split(r",|;|\band\b|\bor\b|\(|\)", chunk):
            for s in lexicon.extract_skills(clause):
                (nice if NICE_CUES.search(clause) else must).append(s)
    must = list(dict.fromkeys(must))
    nice = [s for s in dict.fromkeys(nice) if s not in must]
    return must, nice


class RuleExtractor:
    """Deterministic baseline extractor."""

    def extract(self, text: str, current: JobRequirements, asked: str | None) -> JobRequirements:
        r = current.model_copy(deep=True)
        is_none = bool(NONE_ANSWER.search(text))

        role = lexicon.detect_role(text)
        if role and (r.role is None or asked == "role"):
            r.role = role
            if r.title is None:
                r.title = text.strip()[:80] if asked == "role" else None

        must, nice = split_must_nice(text)
        if asked == "must_have_skills" and not NICE_CUES.search(text):
            must, nice = list(dict.fromkeys(must + nice)), []
        elif r.must_have_skills is None and asked != "must_have_skills" and not MUST_CUES.search(text) and len(must) < 2:
            # "senior Python developer" names the role, it is not yet an explicit must-have list:
            # keep the skills as pending so the collector still asks for must-haves.
            r.pending_skills = list(dict.fromkeys(r.pending_skills + must))
            must = []
        if must:
            r.must_have_skills = list(dict.fromkeys(r.pending_skills + (r.must_have_skills or []) + must))
            r.pending_skills = []
        elif asked == "must_have_skills" and is_none:
            r.must_have_skills = []
        if nice:
            r.nice_to_have_skills = list(
                dict.fromkeys(r.nice_to_have_skills + [s for s in nice if s not in (r.must_have_skills or [])])
            )

        yrs = _years(text, asked)
        if yrs is not None and (r.min_years is None or asked in (None, "min_years")):
            r.min_years = yrs
        elif asked == "min_years" and is_none:
            r.min_years = 0.0

        eng = lexicon.detect_english(text)
        if eng is None and asked == "english_level":
            for pat, level in lexicon.ENGLISH_PATTERNS + [(r"\bupper\b", "upper"), (r"\bpre\b", "pre")]:
                if re.search(pat, text.lower()):
                    eng = level
                    break
            if eng is None and is_none:
                eng = "no_english"
        if eng:
            r.english_level = eng

        sen = lexicon.detect_seniority(text)
        if sen:
            r.seniority = sen
        doms = lexicon.detect_domains(text)
        if doms and not r.domain:
            r.domain = doms[0]
        r.summary = (r.summary + " " + text).strip()[-1500:]
        return r


class _LLMSlots(BaseModel):
    title: str | None = None
    role: str | None = None
    must_have_skills: list[str] | None = None
    nice_to_have_skills: list[str] = []
    min_years: float | None = None
    seniority: str = "unknown"
    english_level: str | None = None
    domain: str | None = None


class LLMExtractor:
    """LLM slot filler; falls back to rules if the LLM is unavailable."""

    def __init__(self) -> None:
        self.fallback = RuleExtractor()

    def extract(self, text: str, current: JobRequirements, asked: str | None) -> JobRequirements:
        if not config.llm_enabled():
            return self.fallback.extract(text, current, asked)
        prompt = llm.load_prompt(
            "c1_extract",
            roles=", ".join(lexicon.ROLES),
            current=current.model_dump_json(exclude={"summary"}),
            asked=asked or "none",
            message=text,
        )
        try:
            slots = llm.complete_json(prompt, _LLMSlots, component="C1", temperature=config.TEMPERATURE_SCORING)
        except (llm.LLMUnavailable, ValidationError, ValueError):
            llm.note(False)
            return self.fallback.extract(text, current, asked)
        llm.note(True)
        r = current.model_copy(deep=True)
        data = slots.model_dump()
        if data["role"] not in lexicon.ROLES:
            data["role"] = lexicon.detect_role(data["role"] or "") or lexicon.detect_role(text) or r.role
        for key in ("must_have_skills", "nice_to_have_skills"):
            if data[key] is not None:
                data[key] = list(dict.fromkeys(lexicon.normalize_skill(s) or s for s in data[key]))
        if data["english_level"] not in config.ENGLISH_ORDER:
            data["english_level"] = r.english_level
        if data["seniority"] not in config.SENIORITY_ORDER:
            data["seniority"] = r.seniority
        for k, v in data.items():
            if v is not None and v != [] or (k == "must_have_skills" and v == [] and asked == k):
                setattr(r, k, v)
        r.summary = (r.summary + " " + text).strip()[-1500:]
        return r


class RequirementCollector:
    """Dialogue policy: ask for the first missing required slot."""

    def __init__(self, extractor: Extractor | None = None) -> None:
        self.extractor = extractor or (LLMExtractor() if config.llm_enabled() else RuleExtractor())
        self.state = JobRequirements()
        self.asked: str | None = None
        self.turns = 0

    def respond(self, user_text: str) -> str:
        self.turns += 1
        self.state = self.extractor.extract(user_text, self.state, self.asked)
        missing = self.state.missing_slots()
        if not missing:
            self.asked = None
            return DONE_MESSAGE
        self.asked = missing[0]
        return QUESTIONS[self.asked]

    @property
    def done(self) -> bool:
        return self.state.is_complete()


REQ_HEADER = re.compile(
    r"requirement|qualification|must|skills|what (?:we|you) (?:expect|need|bring)|you (?:have|should|will need)|"
    r"we(?:'re| are) looking for|tech(?:nical)? stack|\bstack\b|experience|knowledge",
    re.I,
)
OTHER_HEADER = re.compile(
    r"responsibilit|duties|you will|what you(?:'ll| will) do|about|offer|benefit|we provide|project|company|why|perks|tasks",
    re.I,
)
HEADER_LINE = re.compile(r"^\s*(?:#+\s*|\*\*)?([^\n]{2,60}?)(?:\*\*)?\s*:?\s*$")


def _sections(text: str) -> list[tuple[str, str]]:
    """Split a JD into (header, body) by short header-like lines ('Requirements:', '**About us**')."""
    out: list[tuple[str, str]] = [("", "")]
    for line in text.splitlines():
        s = line.strip()
        m = HEADER_LINE.match(s)
        is_header = bool(m) and (s.endswith(":") or s.startswith(("**", "#"))) and len(s) <= 64
        if is_header:
            out.append((m.group(1), ""))
        else:
            out[-1] = (out[-1][0], out[-1][1] + line + "\n")
    return [(h, b) for h, b in out if b.strip() or h]


def requirements_from_jd(jd_text: str, extractor: Extractor | None = None) -> JobRequirements:
    """Single-shot extraction from a full job description (used by evals and the UI).

    Two guards against the errors found by manual labelling (reports/05):
    - the role comes from the job TITLE (first line) when it names one; the body often
      mentions other roles ("our DevOps team…");
    - must-haves come from requirement sections when the JD has them, plus skills named in
      the title, capped at ONESHOT_MAX_MUST; the overflow becomes nice-to-have.
    """
    ex = extractor or RuleExtractor()
    req = ex.extract(jd_text, JobRequirements(), None)
    title = jd_text.strip().split("\n", 1)[0]
    title_role = lexicon.detect_role(title)
    if title_role:
        req.role = title_role
    if isinstance(ex, RuleExtractor):
        secs = _sections(jd_text)
        req_text = "\n".join(b for h, b in secs if h and REQ_HEADER.search(h) and not OTHER_HEADER.search(h))
        if req_text.strip():
            must, nice = split_must_nice(req_text)
            title_skills = lexicon.extract_skills(title)
            must = list(dict.fromkeys(title_skills + must))
            req.must_have_skills = must
            req.nice_to_have_skills = list(dict.fromkeys(nice + [s for s in req.nice_to_have_skills if s not in must]))
    if req.must_have_skills is None:
        req.must_have_skills = list(req.pending_skills)
    if len(req.must_have_skills) > config.ONESHOT_MAX_MUST:
        overflow = req.must_have_skills[config.ONESHOT_MAX_MUST :]
        req.must_have_skills = req.must_have_skills[: config.ONESHOT_MAX_MUST]
        req.nice_to_have_skills = list(dict.fromkeys(req.nice_to_have_skills + overflow))
    req.nice_to_have_skills = [s for s in req.nice_to_have_skills if s not in req.must_have_skills]
    if req.min_years is None:
        req.min_years = 0.0
    if req.english_level is None:
        req.english_level = "no_english"
    req.summary = jd_text[:1500]
    return req
