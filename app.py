"""C7 · Streamlit UI for the Recruiter Assistant.

Run: streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from recruiter import config, llm, pipeline  # noqa: E402
from recruiter.c1_requirements import RequirementCollector, requirements_from_jd  # noqa: E402
from recruiter.schemas import JobRequirements  # noqa: E402

st.set_page_config(page_title="Recruiter Assistant", layout="wide")


def reset() -> None:
    st.session_state.collector = RequirementCollector()
    st.session_state.chat = [("assistant", "Hi! Tell me about the role you're hiring for.")]
    st.session_state.result = None


if "collector" not in st.session_state:
    reset()

with st.sidebar:
    st.header("Recruiter Assistant")
    mode = "LLM (" + config.LLM_MODEL + ")" if config.llm_enabled() else "rule-based (no API key)"
    st.caption(f"Mode: **{mode}**")
    if config.llm_enabled() and llm.STATS["fallbacks"]:
        st.warning(
            f"{llm.STATS['fallbacks']} LLM call(s) failed, so those steps used the rule-based fallback. "
            f"Last error: {llm.LAST_ERROR.get('msg', 'unknown')}"
        )
    st.caption(f"Rubric weights: {config.RUBRIC_WEIGHTS}")
    if st.button("New search"):
        reset()
        st.rerun()
    st.divider()
    st.subheader("Or paste a full job description")
    jd = st.text_area("Job description", height=180, label_visibility="collapsed")
    if st.button("Extract & search") and jd.strip():
        req = requirements_from_jd(jd, st.session_state.collector.extractor)
        st.session_state.collector.state = req
        st.session_state.chat.append(("user", jd[:300] + ("…" if len(jd) > 300 else "")))
        st.session_state.chat.append(("assistant", "Extracted the requirements from the JD. Searching now."))
        st.session_state.result = pipeline.run(req)

st.title("Find and reach the best candidates")
left, right = st.columns([2, 3])

with left:
    st.subheader("1 · Requirements (C1)")
    for role, text in st.session_state.chat:
        st.chat_message(role).write(text)
    if prompt := st.chat_input("Describe the role, or answer the question"):
        c: RequirementCollector = st.session_state.collector
        st.session_state.chat.append(("user", prompt))
        st.session_state.chat.append(("assistant", c.respond(prompt)))
        if c.done:
            with st.spinner("Filtering, retrieving, scoring, writing…"):
                st.session_state.result = pipeline.run(c.state)
        st.rerun()
    req: JobRequirements = st.session_state.collector.state
    st.caption("Collected slots")
    st.json(req.model_dump(exclude={"summary", "pending_skills"}), expanded=False)

with right:
    res: pipeline.PipelineResult | None = st.session_state.result
    if res is None:
        st.info("Once all required slots (role, must-have skills, minimum years, English) are filled, the search starts.")
    else:
        st.subheader("2 · Pipeline trace")
        st.dataframe(pd.DataFrame(res.trace), hide_index=True, width="stretch")
        if res.must_have_coverage < 1:
            st.warning(
                f"Fewer than {config.RERANK_TOP_K} candidates met every must-have, so the filter was relaxed "
                f"to {res.must_have_coverage:.0%} must-have coverage."
            )
        with st.expander("Why candidates were filtered out (C3)"):
            st.write(res.rejections)

        st.subheader("3 · Ranked shortlist (C5: rubric blended with retrieval)")
        st.caption(
            f"final = {config.RERANK_BLEND_ALPHA} × rubric + {1 - config.RERANK_BLEND_ALPHA:.1f} × retrieval (normalised per search)"
        )
        rows = []
        for i, r in enumerate(res.ranked, 1):
            p = res.profiles[r.candidate_id]
            rows.append(
                {
                    "#": i,
                    "final": round(r.final, 3),
                    "rubric": round(r.total, 3),
                    "title": p.title[:60],
                    "years": "10+"
                    if (p.years_experience or 0) >= 11
                    else f"{p.years_experience:g}"
                    if p.years_experience is not None
                    else "?",
                    "English": p.english_level,
                    "retrieval rank": r.retrieval_rank,
                    **{c.criterion: c.score for c in r.criteria},
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        for i, r in enumerate(res.ranked[:5], 1):
            p = res.profiles[r.candidate_id]
            with st.expander(f"#{i} · {p.title[:70]} · final {r.final:.2f}, rubric {r.total:.2f}: evidence"):
                for c in r.criteria:
                    st.markdown(f"**{c.criterion}**: {c.score}/5 · {c.rationale}")
                    if c.evidence:
                        st.caption(f"“{c.evidence}”")
                st.caption("CV excerpt")
                st.text(p.cv_text[:1200])

        st.subheader("4 · Outreach messages (C6, verified)")
        for m in res.messages:
            p = res.profiles[m.candidate_id]
            badge = "✅ all claims supported" if m.faithful else "⚠️ unsupported: " + "; ".join(m.unsupported_claims)
            st.markdown(f"**{p.title[:70]}**: {m.writer} · {m.claims_checked} claims checked · {badge}")
            st.text_area("message", m.text, height=260, key=f"msg_{m.candidate_id}", label_visibility="collapsed")

        log = config.LOG_DIR / "llm_calls.jsonl"
        if config.llm_enabled() and log.exists():
            with st.expander("LLM call log (last 20)"):
                lines = log.read_text().splitlines()[-20:]
                st.dataframe(pd.DataFrame([json.loads(x) for x in lines]))
