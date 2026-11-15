"""fit_labels as a shared eval bed for C3/C4/C5: JD -> JobRequirements (C1 single-shot),
resume -> CandidateProfile (C2 free-text path). Ids are stable content hashes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from common import load_fit
from recruiter.c1_requirements import requirements_from_jd
from recruiter.c2_profiles import profile_from_text
from recruiter.c4_retrieval import HybridIndex
from recruiter.schemas import CandidateProfile, JobRequirements


def hid(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


@dataclass
class FitBed:
    pairs: pd.DataFrame  # jd_id, cv_id, label, gain
    reqs: dict[str, JobRequirements]
    jd_text: dict[str, str]
    profiles: dict[str, CandidateProfile]


@lru_cache(maxsize=2)
def bed(split: str = "test") -> FitBed:
    df = load_fit(split)
    df["jd_id"] = df["job_description_text"].map(hid)
    df["cv_id"] = df["resume_text"].map(hid)
    jd_text = dict(zip(df["jd_id"], df["job_description_text"]))
    reqs = {j: requirements_from_jd(t) for j, t in jd_text.items()}
    cvs = dict(zip(df["cv_id"], df["resume_text"]))
    profiles = {c: profile_from_text(c, t) for c, t in cvs.items()}
    return FitBed(df[["jd_id", "cv_id", "label", "gain"]].drop_duplicates(["jd_id", "cv_id"]), reqs, jd_text, profiles)


@lru_cache(maxsize=2)
def index(split: str = "test") -> HybridIndex:
    b = bed(split)
    ids = sorted(b.profiles)
    return HybridIndex.build(ids, [b.profiles[i].cv_text for i in ids], dim=128)
