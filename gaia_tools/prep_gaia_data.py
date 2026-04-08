#!/usr/bin/env python3
"""Fetch full GAIA questions (all levels) and download attached files.

Called automatically by setup.sh and by ./ask if GAIA data is missing.

Fetches GAIA 2023 from the gated HuggingFace dataset:
  - test split (301 questions, no answers) → gaia_questions.json (benchmark)
  - validation split (165 questions, with answers) → gaia_dev_questions.json (tuning)

Falls back to the HF course API (Level 1 only) if the gated dataset is inaccessible.

Prerequisites:
    - HF_TOKEN with access to the gated dataset: https://huggingface.co/datasets/gaia-benchmark/GAIA
    - pip install datasets huggingface_hub requests

Produces:
    gaia_questions.json       - test questions (no answers) for benchmarking
    gaia_dev_questions.json   - dev/validation questions (with answers) for tuning
    gaia_files/               - all attached files (images, audio, spreadsheets, etc.)
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


def _normalize_rows(ds, include_answers=True):
    """Convert HF dataset rows to normalized dicts."""
    questions = []
    for row in ds:
        q = {
            "task_id": row.get("task_id", ""),
            "question": row.get("Question", row.get("question", "")),
            "level": str(row.get("Level", row.get("level", "1"))),
            "file_name": row.get("file_name", ""),
        }
        if include_answers:
            q["Final answer"] = row.get("Final answer", "")
        questions.append(q)
    return questions


def _fetch_full_dataset(token):
    """Fetch GAIA test + validation splits from the HuggingFace dataset.

    Returns (test_questions, dev_questions) or (None, None) on failure.
    Test questions have no answers; dev questions have expected answers.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  'datasets' package not installed. Install it with:")
        print("    pip install datasets")
        return None, None

    if not token:
        print("  No HF token. Cannot access the gated GAIA dataset.")
        return None, None

    print("Fetching GAIA dataset from HuggingFace (all levels)...")
    try:
        ds_test = load_dataset(
            GAIA_HF_REPO, "2023_all", split="test", token=token
        )
        ds_val = load_dataset(
            GAIA_HF_REPO, "2023_all", split="validation", token=token
        )
    except Exception as e:
        print(f"  Failed to load dataset: {e}")
        print("  Make sure you accepted the terms at:")
        print(f"  https://huggingface.co/datasets/{GAIA_HF_REPO}")
        return None, None

    test_questions = _normalize_rows(ds_test, include_answers=False)
    dev_questions = _normalize_rows(ds_val, include_answers=True)

    for label, qs in [("Test", test_questions), ("Dev", dev_questions)]:
        levels = Counter(q["level"] for q in qs)
        print(f"  {label}: {len(qs)} questions — "
              + ", ".join(f"L{k}={v}" for k, v in sorted(levels.items())))
    return test_questions, dev_questions


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
    dev_json = Path("gaia_dev_questions.json")
    files_dir = Path("gaia_files")
    files_dir.mkdir(exist_ok=True)

    token = _get_token()

    test_questions, dev_questions = _fetch_full_dataset(token)

    # Fallback: if full dataset failed, use course API for test questions only
    if test_questions is None:
        print("\nFalling back to HF course API (Level 1 only)...")
        test_questions = _fetch_course_api()
        dev_questions = None

    if not test_questions:
        print("ERROR: Could not fetch questions from any source.")
        sys.exit(1)

    out_json.write_text(json.dumps(test_questions, indent=2))
    print(f"\nSaved {len(test_questions)} test questions to {out_json}")

    if dev_questions:
        dev_json.write_text(json.dumps(dev_questions, indent=2))
        print(f"Saved {len(dev_questions)} dev questions to {dev_json}")

    all_questions = test_questions + (dev_questions or [])
    needed = [q.get("file_name", "") for q in all_questions if q.get("file_name")]
    missing = [fn for fn in needed if not (files_dir / fn).exists()]
    print(f"\nAttached files: {len(needed)} total, {len(needed) - len(missing)} cached, "
          f"{len(missing)} to download.")

    if not missing:
        print("All files already present.")
        _print_summary(test_questions, dev_questions)
        return

    if not token:
        print("ERROR: No HF token found. Set HF_TOKEN or run `huggingface-cli login`.")
        print(f"Token must have access to https://huggingface.co/datasets/{GAIA_HF_REPO}")
        return

    for subdir, qs in [("2023/test", test_questions), (GAIA_HF_SUBDIR, dev_questions or [])]:
        for i, fn in enumerate(
            [q["file_name"] for q in qs if q.get("file_name") and not (files_dir / q["file_name"]).exists()],
            1,
        ):
            dest = files_dir / fn
            try:
                downloaded = hf_hub_download(
                    repo_id=GAIA_HF_REPO,
                    filename=f"{subdir}/{fn}",
                    repo_type="dataset",
                    token=token,
                )
                shutil.copy2(downloaded, dest)
                print(f"  [{i}] {fn}")
            except Exception as e:
                print(f"  [{i}] FAILED {fn}: {e}")

    _print_summary(test_questions, dev_questions)
    print("\nDone. GAIA data is ready.")


def _print_summary(test_questions, dev_questions=None):
    """Print a summary of the question sets."""
    print(f"\nSummary:")
    for label, qs in [("Test (benchmark)", test_questions), ("Dev (with answers)", dev_questions)]:
        if not qs:
            continue
        levels = Counter(str(q.get("level", "?")) for q in qs)
        has_files = sum(1 for q in qs if q.get("file_name"))
        print(f"  {label}: {len(qs)} questions "
              + ", ".join(f"L{k}={v}" for k, v in sorted(levels.items()))
              + f", {has_files} with files")


if __name__ == "__main__":
    main()
