#!/usr/bin/env python3
"""Fetch full GAIA questions (all levels) and download attached files.

Called automatically by setup.sh and by ./ask if GAIA data is missing.

Fetches the complete GAIA 2023 validation set from the gated HuggingFace dataset
(Levels 1, 2, and 3). Falls back to the HF course API (Level 1 only) if the gated
dataset is inaccessible.

Prerequisites:
    - HF_TOKEN with access to the gated dataset: https://huggingface.co/datasets/gaia-benchmark/GAIA
    - pip install datasets huggingface_hub requests

Produces:
    gaia_questions.json   - all validation questions (L1 + L2 + L3) with expected answers
    gaia_files/           - all attached files (images, audio, spreadsheets, etc.)
"""

import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

API_URL = "https://agents-course-unit4-scoring.hf.space"
GAIA_HF_REPO = "gaia-benchmark/GAIA"
GAIA_HF_SUBDIR = "2023/validation"


def _get_token():
    """Get HF token from env or cached login."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token() or ""
        except Exception:
            pass
    return token


def _fetch_full_dataset(token):
    """Fetch all GAIA validation questions (L1+L2+L3) from the HuggingFace dataset.

    Returns a list of dicts with normalized field names, or None on failure.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  'datasets' package not installed. Install it with:")
        print("    pip install datasets")
        return None

    if not token:
        print("  No HF token. Cannot access the gated GAIA dataset.")
        return None

    print("Fetching full GAIA dataset from HuggingFace (all levels)...")
    try:
        ds = load_dataset(
            GAIA_HF_REPO, "2023_all", split="validation", token=token
        )
    except Exception as e:
        print(f"  Failed to load dataset: {e}")
        print("  Make sure you accepted the terms at:")
        print(f"  https://huggingface.co/datasets/{GAIA_HF_REPO}")
        return None

    questions = []
    for row in ds:
        q = {
            "task_id": row.get("task_id", ""),
            "question": row.get("Question", row.get("question", "")),
            "level": str(row.get("Level", row.get("level", "1"))),
            "Final answer": row.get("Final answer", ""),
            "file_name": row.get("file_name", ""),
        }
        questions.append(q)

    levels = Counter(q["level"] for q in questions)
    print(f"  Got {len(questions)} questions: "
          + ", ".join(f"L{k}={v}" for k, v in sorted(levels.items())))
    return questions


def _fetch_course_api():
    """Fetch questions from the HF course scoring API (Level 1 only, no answers)."""
    print("Fetching questions from HF course scoring API (Level 1 only)...")
    try:
        resp = requests.get(f"{API_URL}/questions", timeout=30)
        resp.raise_for_status()
        questions = resp.json()
        for q in questions:
            if "Level" in q and "level" not in q:
                q["level"] = str(q["Level"])
            if "question" not in q and "Question" in q:
                q["question"] = q["Question"]
        print(f"  Got {len(questions)} questions (Level 1 only, no expected answers).")
        return questions
    except Exception as e:
        print(f"  HF course API failed: {e}")
        return None


def main():
    out_json = Path("gaia_questions.json")
    files_dir = Path("gaia_files")
    files_dir.mkdir(exist_ok=True)

    token = _get_token()

    questions = _fetch_full_dataset(token)
    if questions is None:
        print("\nFalling back to HF course API (Level 1 only)...")
        questions = _fetch_course_api()

    if not questions:
        print("ERROR: Could not fetch questions from any source.")
        sys.exit(1)

    out_json.write_text(json.dumps(questions, indent=2))
    print(f"\nSaved {len(questions)} questions to {out_json}")

    needed = [q.get("file_name", "") for q in questions if q.get("file_name")]
    missing = [fn for fn in needed if not (files_dir / fn).exists()]
    print(f"\nAttached files: {len(needed)} total, {len(needed) - len(missing)} cached, "
          f"{len(missing)} to download.")

    if not missing:
        print("All files already present.")
        _print_summary(questions)
        return

    if not token:
        print("ERROR: No HF token found. Set HF_TOKEN or run `huggingface-cli login`.")
        print(f"Token must have access to https://huggingface.co/datasets/{GAIA_HF_REPO}")
        return

    for i, fn in enumerate(missing, 1):
        dest = files_dir / fn
        try:
            downloaded = hf_hub_download(
                repo_id=GAIA_HF_REPO,
                filename=f"{GAIA_HF_SUBDIR}/{fn}",
                repo_type="dataset",
                token=token,
            )
            shutil.copy2(downloaded, dest)
            print(f"  [{i}/{len(missing)}] {fn}")
        except Exception as e:
            print(f"  [{i}/{len(missing)}] FAILED {fn}: {e}")

    _print_summary(questions)
    print("\nDone. GAIA data is ready in gaia_questions.json and gaia_files/.")


def _print_summary(questions):
    """Print a summary of the question set."""
    levels = Counter(str(q.get("level", q.get("Level", "?"))) for q in questions)
    has_answers = sum(1 for q in questions if q.get("Final answer"))
    has_files = sum(1 for q in questions if q.get("file_name"))
    print(f"\nSummary:")
    print(f"  Total questions: {len(questions)}")
    for lvl in sorted(levels):
        print(f"  Level {lvl}: {levels[lvl]} questions")
    print(f"  With expected answers: {has_answers}")
    print(f"  With attached files: {has_files}")


if __name__ == "__main__":
    main()
