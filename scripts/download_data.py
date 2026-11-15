"""Download the datasets used by the Recruiter Assistant into data/raw/.

Usage (from repo root):
    pip install datasets huggingface_hub pandas pyarrow
    python scripts/download_data.py            # everything
    python scripts/download_data.py --only djinni_cvs djinni_jobs

data/raw/ is git-ignored: never commit the datasets, commit this script instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# name -> (Hugging Face repo id, purpose)
HF_TABLES = {
    "djinni_cvs": (
        "lang-uk/recruitment-dataset-candidate-profiles-english",
        "Candidate database (real, anonymized Djinni profiles, MIT)",
    ),
    "djinni_jobs": (
        "lang-uk/recruitment-dataset-job-descriptions-english",
        "Job postings: test inputs for requirement collection, MIT",
    ),
    "fit_labels": (
        "cnamuangtoun/resume-job-description-fit",
        "Eval: resume/JD pairs labelled No/Potential/Good Fit",
    ),
}

# Raw file repos (not loadable as a table) -> snapshot download
HF_SNAPSHOTS = {
    "skills_ner": (
        "Mehyaar/Annotated_NER_PDF_Resumes",
        "Eval: 5,029 CVs with IT-skill span annotations (extraction F1)",
    ),
}


def download_table(name: str, repo: str) -> None:
    from datasets import load_dataset

    out = RAW / name
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(repo)
    for split, part in ds.items():
        path = out / f"{split}.parquet"
        part.to_parquet(path)
        print(f"  {name}/{split}: {part.num_rows:,} rows -> {path}")
        print(f"    columns: {part.column_names}")


def download_snapshot(name: str, repo: str) -> None:
    from huggingface_hub import snapshot_download

    out = RAW / name
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=out)
    print(f"  {name}: files -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="subset of dataset names")
    args = parser.parse_args()
    wanted = set(args.only) if args.only else None

    for name, (repo, purpose) in {**HF_TABLES, **HF_SNAPSHOTS}.items():
        if wanted and name not in wanted:
            continue
        print(f"[{name}] {repo} :: {purpose}")
        try:
            if name in HF_TABLES:
                download_table(name, repo)
            else:
                download_snapshot(name, repo)
        except Exception as exc:  # keep going if one source fails
            print(f"  FAILED: {exc}")

    # TalentCLEF was planned for ranking eval; replaced by fit_labels test (graded qrels), see reports/02.


if __name__ == "__main__":
    main()
