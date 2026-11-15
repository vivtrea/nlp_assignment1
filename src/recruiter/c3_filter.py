"""C3 · Hard filter: deterministic business rules. Failing a rule removes the candidate.

Rules (each yields a human-readable reason when it fails):
1. role      : candidate role in the related-role set of the job role (skipped if either unknown)
2. must-have : share of must-have skills found >= FILTER_MUST_HAVE_MIN_COVERAGE
3. years     : years_experience >= min_years - tolerance (unknown passes if configured)
4. english   : english_level >= required (unknown passes if configured)
5. level     : for junior jobs, no senior/lead candidates (FILTER_OVERQUALIFIED_*)
"""

from __future__ import annotations

from recruiter import config, lexicon
from recruiter.schemas import CandidateProfile, FilterDecision, JobRequirements


def _eng_rank(level: str | None) -> int | None:
    return config.ENGLISH_ORDER.index(level) if level in config.ENGLISH_ORDER else None


def has_skill(profile: CandidateProfile, skill: str) -> bool:
    canon = lexicon.normalize_skill(skill)
    if canon is not None:
        return canon in profile.skills  # skills were extracted from the same text by C2
    return lexicon.find_skill_span(profile.evidence_text(), skill) is not None


def check(
    req: JobRequirements,
    p: CandidateProfile,
    *,
    use_role: bool = True,
    min_coverage: float = config.FILTER_MUST_HAVE_MIN_COVERAGE,
) -> FilterDecision:
    reasons: list[str] = []
    if use_role and req.role and p.role and p.role not in lexicon.related_roles(req.role):
        reasons.append(f"role {p.role} not related to {req.role}")

    must = req.must_have_skills or []
    if must:
        missing = [s for s in must if not has_skill(p, s)]
        coverage = 1 - len(missing) / len(must)
        if coverage + 1e-9 < min_coverage:
            reasons.append(f"missing must-have: {', '.join(missing)}")

    if req.min_years:
        if p.years_experience is None:
            if not config.FILTER_UNKNOWN_YEARS_PASSES:
                reasons.append("years unknown")
        elif p.years_experience < req.min_years - config.FILTER_YEARS_TOLERANCE:
            reasons.append(f"{p.years_experience:g}y < required {req.min_years:g}y")

    if req.seniority in config.FILTER_OVERQUALIFIED_FOR and p.seniority in config.SENIORITY_ORDER:
        gap = config.SENIORITY_ORDER.index(p.seniority) - config.SENIORITY_ORDER.index(req.seniority)
        if gap >= config.FILTER_OVERQUALIFIED_GAP:
            reasons.append(f"over-qualified: {p.seniority} for a {req.seniority} role")

    need = _eng_rank(req.english_level)
    if need:
        have = _eng_rank(p.english_level)
        if have is None:
            if not config.FILTER_UNKNOWN_ENGLISH_PASSES:
                reasons.append("English unknown")
        elif have < need:
            reasons.append(f"English {p.english_level} < {req.english_level}")

    return FilterDecision(candidate_id=p.candidate_id, passed=not reasons, reasons=reasons)


def apply(
    req: JobRequirements,
    profiles: list[CandidateProfile],
    *,
    use_role: bool = True,
    min_coverage: float = config.FILTER_MUST_HAVE_MIN_COVERAGE,
) -> tuple[list[CandidateProfile], list[FilterDecision]]:
    decisions = [check(req, p, use_role=use_role, min_coverage=min_coverage) for p in profiles]
    kept = [p for p, d in zip(profiles, decisions) if d.passed]
    return kept, decisions
