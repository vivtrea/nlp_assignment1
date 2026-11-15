"""Pydantic models for everything that crosses a component boundary."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EnglishLevel = Literal["no_english", "basic", "pre", "intermediate", "upper", "fluent", "unknown"]
Seniority = Literal["junior", "middle", "senior", "lead", "unknown"]

REQUIRED_SLOTS = ("role", "must_have_skills", "min_years", "english_level")


class JobRequirements(BaseModel):
    """C1 output. `None` = not collected yet; explicit 'any' values mean 'no requirement'."""

    title: str | None = None
    role: str | None = Field(None, description="Djinni Primary Keyword, e.g. 'Python'")
    must_have_skills: list[str] | None = None
    nice_to_have_skills: list[str] = Field(default_factory=list)
    min_years: float | None = None
    seniority: Seniority = "unknown"
    english_level: EnglishLevel | None = None  # "no_english" = no requirement
    domain: str | None = None
    summary: str = ""
    pending_skills: list[str] = Field(default_factory=list, description="skills seen before must-haves were asked")

    def missing_slots(self) -> list[str]:
        return [s for s in REQUIRED_SLOTS if getattr(self, s) is None]

    def is_complete(self) -> bool:
        return not self.missing_slots()

    def query_text(self) -> str:
        parts = [self.title or "", self.role or "", self.seniority if self.seniority != "unknown" else ""]
        parts += (self.must_have_skills or []) * 2 + self.nice_to_have_skills
        parts += [self.domain or "", self.summary]
        return " ".join(p for p in parts if p)


class CandidateProfile(BaseModel):
    """C2 output."""

    candidate_id: str
    title: str = ""
    role: str | None = None
    skills: list[str] = Field(default_factory=list)
    years_experience: float | None = None
    seniority: Seniority = "unknown"
    english_level: EnglishLevel = "unknown"
    domains: list[str] = Field(default_factory=list)
    looking_for: str = ""
    cv_text: str = ""

    def evidence_text(self) -> str:
        return f"{self.title}\n{self.cv_text}\n{self.looking_for}"


class CriterionScore(BaseModel):
    criterion: str
    score: int = Field(ge=1, le=5)
    evidence: str = ""
    rationale: str = ""


class RankedCandidate(BaseModel):
    candidate_id: str
    total: float = Field(ge=0.0, le=1.0)
    criteria: list[CriterionScore]
    retrieval_rank: int | None = None
    retrieval_score: float | None = None
    final: float = Field(0.0, ge=0.0, le=1.0, description="blend of rubric total and retrieval score; sort key")
    scorer: Literal["rules", "llm"] = "rules"


class OutreachMessage(BaseModel):
    candidate_id: str
    text: str
    writer: Literal["template", "llm"] = "template"
    claims_checked: int = 0
    unsupported_claims: list[str] = Field(default_factory=list)

    @property
    def faithful(self) -> bool:
        return not self.unsupported_claims


class FilterDecision(BaseModel):
    candidate_id: str
    passed: bool
    reasons: list[str] = Field(default_factory=list)
