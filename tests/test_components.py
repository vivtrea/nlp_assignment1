"""Unit tests for component contracts and business rules (no data files, no LLM)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recruiter import config, lexicon  # noqa: E402
from recruiter.c1_requirements import RequirementCollector, RuleExtractor, split_must_nice  # noqa: E402
from recruiter.c3_filter import check  # noqa: E402
from recruiter.c4_retrieval import HybridIndex, rrf  # noqa: E402
from recruiter.c5_reranker import RuleScorer, total_score, verify_evidence  # noqa: E402
from recruiter.c6_outreach import TemplateWriter, verify_claims  # noqa: E402
from recruiter.schemas import CandidateProfile, CriterionScore, JobRequirements  # noqa: E402


def profile(**kw) -> CandidateProfile:
    base = dict(
        candidate_id="c1",
        title="Python Developer",
        role="Python",
        skills=["Python", "Django", "Docker"],
        years_experience=5.0,
        seniority="senior",
        english_level="upper",
        looking_for="Product company, Django and Python backend work",
        cv_text="Built REST services with Django and Python. Deployed with Docker.",
    )
    base.update(kw)
    return CandidateProfile(**base)


REQ = JobRequirements(
    role="Python",
    must_have_skills=["Python", "Django"],
    nice_to_have_skills=["AWS"],
    min_years=3,
    english_level="intermediate",
    seniority="senior",
    summary="Python Django backend",
)


def test_weights_sum_to_one():
    assert abs(sum(config.RUBRIC_WEIGHTS.values()) - 1) < 1e-9


def test_lexicon_boundaries():
    assert lexicon.extract_skills("Java and JavaScript") == ["Java", "JavaScript"]
    assert "Go" not in lexicon.extract_skills("let's go to google")
    assert "SQL" not in lexicon.extract_skills("MS SQL only")
    assert "Spring" not in lexicon.extract_skills("spring cleaning")


def test_must_nice_split_per_clause():
    must, nice = split_must_nice("React, TypeScript required, Next.js is a plus.")
    assert must == ["React", "TypeScript"] and nice == ["Next.js"]


def test_collector_asks_until_complete():
    c = RequirementCollector(RuleExtractor())
    c.respond("We need a senior Python developer")
    assert c.asked == "must_have_skills"
    c.respond("Django and PostgreSQL")
    assert c.asked == "min_years"
    c.respond("4 years")
    c.respond("no requirement")
    assert c.done and c.state.english_level == "no_english" and c.state.min_years == 4


def test_filter_rules():
    assert check(REQ, profile()).passed
    assert not check(REQ, profile(skills=["Python"])).passed  # missing Django
    assert not check(REQ, profile(years_experience=1.0)).passed
    assert not check(REQ, profile(english_level="pre")).passed
    assert not check(REQ, profile(role="Design")).passed
    assert check(REQ, profile(english_level="unknown")).passed  # unknown passes by config


def test_no_evidence_scores_one():
    crit = {c.criterion: c for c in RuleScorer().score(REQ, profile(looking_for=""))}
    assert crit["motivation"].score == 1
    fake = CriterionScore(criterion="must_have_skills", score=5, evidence="led a team of 40 at Google")
    assert verify_evidence(fake, profile()).score == 1


def test_overqualified_capped():
    junior = REQ.model_copy(update={"seniority": "junior", "min_years": 0})
    crit = {c.criterion: c for c in RuleScorer().score(junior, profile(seniority="lead", years_experience=11.0))}
    assert crit["relevant_experience"].score <= 3


def test_total_in_range():
    t = total_score(RuleScorer().score(REQ, profile()))
    assert 0 <= t <= 1


def test_verifier_catches_hallucinations():
    p = profile()
    clean = TemplateWriter().write(REQ, p)
    assert verify_claims(clean, p, REQ)[1] == []
    assert verify_claims(clean + "\nYour Kafka expertise is great.", p, REQ)[1]
    assert verify_claims(clean + "\nYou have 12 years of experience.", p, REQ)[1]
    assert verify_claims(clean + '\nYou wrote "scaled payments to 1M users".', p, REQ)[1]


def test_rrf_and_index():
    assert rrf([[0, 1], [1, 0]])[0] == rrf([[0, 1], [1, 0]])[1]
    docs = [
        "python django backend developer",
        "react frontend typescript",
        "java spring kafka",
        "python data science pandas",
        "devops kubernetes terraform",
    ]
    idx = HybridIndex.build([f"d{i}" for i in range(5)], docs, dim=3)
    assert idx.search("django python", top_n=1)[0][0] == "d0"
    assert idx.search("django python", top_n=5, restrict_to=["d1", "d2"])[0][0] in {"d1", "d2"}


def test_blend_orders_by_final():
    from recruiter.c5_reranker import blend_and_sort
    from recruiter.schemas import RankedCandidate

    crit = RuleScorer().score(REQ, profile())
    a = RankedCandidate(candidate_id="a", total=0.9, criteria=crit, retrieval_rank=3, retrieval_score=0.01)
    b = RankedCandidate(candidate_id="b", total=0.5, criteria=crit, retrieval_rank=1, retrieval_score=0.05)
    assert [c.candidate_id for c in blend_and_sort([a, b], alpha=1.0)] == ["a", "b"]
    assert [c.candidate_id for c in blend_and_sort([a, b], alpha=0.0)] == ["b", "a"]
    assert all(0 <= c.final <= 1 for c in blend_and_sort([a, b]))


def test_two_stage_llm_only_rescored_top(monkeypatch):
    from recruiter import c5_reranker
    from recruiter.c5_reranker import LLMScorer, rerank

    calls = []

    class FakeLLM(LLMScorer):
        def score(self, req, p):
            calls.append(p.candidate_id)
            return RuleScorer().score(req, p)

    monkeypatch.setattr(c5_reranker.config, "llm_enabled", lambda: True)
    profs = [profile(candidate_id=f"c{i}", years_experience=float(i)) for i in range(15)]
    retrieval = {p.candidate_id: (i + 1, 1.0 / (i + 1)) for i, p in enumerate(profs)}
    out = rerank(REQ, profs, scorer=FakeLLM(), retrieval=retrieval, llm_top=10)
    assert len(calls) == 10 and len(out) == 15
    assert set(calls) == {c.candidate_id for c in out[:10]}
    assert all(c.scorer == "llm" for c in out[:10]) and all(c.scorer == "rules" for c in out[10:])


def test_oneshot_role_from_title_and_must_cap():
    from recruiter.c1_requirements import requirements_from_jd

    jd = (
        "Junior QA Manual\nOur DevOps team supports the platform.\n"
        "Requirements:\n- Jira, test cases, Postman\n"
        "About us:\nWe use Python, Django, React, Kubernetes, AWS, Terraform, Kafka, Redis."
    )
    req = requirements_from_jd(jd)
    assert req.role == "QA"
    assert "Kubernetes" not in req.must_have_skills and "Jira" in req.must_have_skills
    assert len(req.must_have_skills) <= config.ONESHOT_MAX_MUST


def test_quotes_keep_decimals():
    from recruiter.c5_reranker import sentence_with

    t = "I am a Senior Full-Stack JavaScript Engineer with 4.5 years of experience. Next sentence."
    s, e = t.index("JavaScript"), t.index("JavaScript") + 10
    assert sentence_with(t, (s, e)) == "I am a Senior Full-Stack JavaScript Engineer with 4.5 years of experience"


def test_negative_looking_for_not_used_as_match():
    p = profile(looking_for="No gambling, No part-time.")
    assert "No gambling" not in TemplateWriter().write(REQ, p)
    p2 = profile(looking_for="Considering remote offers and GMT+3 business hours only. I want to grow in backend development.")
    msg = TemplateWriter().write(REQ, p2)
    assert "GMT+3" not in msg and "grow in backend development" in msg


def test_job_type_role_beats_language():
    assert lexicon.detect_role("Middle/Senior AQA Python") == "QA Automation"
    assert lexicon.detect_role("Data Engineer (Python, Spark)") == "Data Engineer"
    assert lexicon.detect_role("Senior Python developer") == "Python"


def test_list_headers_and_non_goals_not_quoted():
    p = profile(
        looking_for="I ll be happy to be a part of:\n- mobile manual testing is pretty routine.\n- a product team where I can grow.",
        cv_text="My achievements include the following:\nBuilt REST services with Django and Python for a fintech product.",
    )
    msg = TemplateWriter().write(REQ, p)
    assert "include the following" not in msg and "be a part of:" not in msg and "pretty routine" not in msg
    assert "a product team where I can grow" in msg


def test_junior_job_rejects_leads():
    junior = REQ.model_copy(update={"seniority": "junior", "min_years": 0})
    assert not check(junior, profile(seniority="lead")).passed
    assert not check(junior, profile(seniority="senior")).passed
    assert check(junior, profile(seniority="middle")).passed
    assert check(REQ, profile(seniority="lead")).passed  # rule only applies to junior jobs
