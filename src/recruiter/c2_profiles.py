"""C2 · Profile extractor: raw CV (+ Djinni form fields) -> CandidateProfile.

Structured Djinni fields are trusted as-is (role, English, years). Skills, seniority and
domains are extracted from text with the lexicon (baseline) or an LLM (optional).
For free-text resumes without form fields (fit_labels), years/English are parsed from text.
"""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, ValidationError

from recruiter import config, lexicon, llm
from recruiter.schemas import CandidateProfile

YEARS_EXP_RE = re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:\w+\s+){0,3}?experience", re.I)
YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\s*(?:-|–|to)\s*((?:19|20)\d{2}|present|current|now|till date|date)\b", re.I)


def seniority_from(years: float | None, title: str) -> str:
    sen = lexicon.detect_seniority(title or "")
    if sen:
        return sen
    if years is None:
        return "unknown"
    if years < 2:
        return "junior"
    if years < 5:
        return "middle"
    if years < 8:
        return "senior"
    return "lead"


def years_from_text(text: str) -> float | None:
    """Explicit 'N years of experience' wins; otherwise span of date ranges (capped 30)."""
    m = YEARS_EXP_RE.findall(text)
    if m:
        return max(float(x) for x in m if float(x) <= 40) if any(float(x) <= 40 for x in m) else None
    ranges = YEAR_RANGE_RE.findall(text)
    if not ranges:
        return None
    this_year = date.today().year
    starts = [int(a) for a, _ in ranges]
    ends = [this_year if not b.isdigit() else int(b) for _, b in ranges]
    span = max(ends) - min(starts)
    return float(min(max(span, 0), 30))


def _clean(text: str | None) -> str:
    return re.sub(r"\r\n?", "\n", text or "").strip()


def profile_from_djinni(row: dict) -> CandidateProfile:
    cv = _clean(row.get("CV"))
    title = _clean(row.get("Position"))
    years = row.get("Experience Years")
    years = float(years) if years is not None else None
    eng = row.get("English Level") or "unknown"
    looking = _clean(row.get("Looking For"))
    if len(looking) < 5 or not re.search(r"[A-Za-z]{3,}", looking):
        looking = ""  # junk values like "1" or "-" carry no motivation signal
    text = f"{title}\n{cv}"
    return CandidateProfile(
        candidate_id=str(row["id"]),
        title=title,
        role=row.get("Primary Keyword"),
        skills=lexicon.extract_skills(text),
        years_experience=years,
        seniority=seniority_from(years, title),
        english_level=eng if eng in config.ENGLISH_ORDER else "unknown",
        domains=lexicon.detect_domains(text + "\n" + looking),
        looking_for=looking,
        cv_text=cv,
    )


def profile_from_text(candidate_id: str, text: str) -> CandidateProfile:
    """For resumes with no structured fields (fit_labels)."""
    text = _clean(text)
    first_line = text.split("\n", 1)[0][:120]
    years = years_from_text(text)
    return CandidateProfile(
        candidate_id=candidate_id,
        title=first_line,
        role=lexicon.detect_role(text[:600]),
        skills=lexicon.extract_skills(text),
        years_experience=years,
        seniority=seniority_from(years, first_line),
        english_level=lexicon.detect_english(text) or "unknown",
        domains=lexicon.detect_domains(text),
        looking_for="",
        cv_text=text,
    )


class _LLMSkills(BaseModel):
    skills: list[str]


def llm_skills(text: str) -> list[str]:
    """LLM skill extraction (C2 'LLM' variant). Only keeps skills literally present in text."""
    prompt = llm.load_prompt("c2_skills", cv=text[:6000])
    try:
        out = llm.complete_json(prompt, _LLMSkills, component="C2", temperature=config.TEMPERATURE_SCORING)
    except (llm.LLMUnavailable, ValidationError, ValueError):
        llm.note(False)
        return lexicon.extract_skills(text)
    llm.note(True)
    low = text.lower()
    return list(dict.fromkeys(s.strip() for s in out.skills if s.strip() and s.strip().lower() in low))
