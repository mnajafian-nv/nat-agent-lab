#!/usr/bin/env python3
"""GAIA benchmark runner with per-level scoring, answer post-processing, and retry logic.

Usage:
    python3 gaia_submit.py NAT-<org>-<team>              # full run
    python3 gaia_submit.py NAT-<org>-<team> --level 1    # Level 1 only (quick test)
    python3 gaia_submit.py NAT-<org>-<team> --local      # local dataset, no HF submit
    python3 gaia_submit.py NAT-<org>-<team> --limit 5    # first 5 questions only
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

API_URL = "https://agents-course-unit4-scoring.hf.space"
NAT_URL = "http://localhost:8000/v1/chat/completions"
LOCAL_RESULTS = Path("gaia_results.json")
LEVEL_WEIGHTS = {1: 1, 2: 2, 3: 3}


_RATE_LIMIT_STRINGS = ["rate limit", "rate_limit", "too many requests", "429", "quota exceeded"]


def ask_nat(question: str, timeout: int = 300) -> str:
    """Send a question to the NAT serve endpoint and get the answer."""
    try:
        resp = requests.post(NAT_URL, json={
            "messages": [{"role": "user", "content": question}],
        }, timeout=timeout)
        if resp.status_code == 429:
            return "FAILED_RATE_LIMIT"
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            err_detail = data.get("error", data.get("detail", ""))
            if isinstance(err_detail, str) and any(s in err_detail.lower() for s in _RATE_LIMIT_STRINGS):
                return "FAILED_RATE_LIMIT"
            return "FAILED: NAT returned empty response (no choices)."
        content = choices[0].get("message", {}).get("content")
        if content is None:
            return "FAILED: NAT returned null content."
        content_lower = content.lower()
        if any(s in content_lower for s in _RATE_LIMIT_STRINGS) and len(content) < 200:
            return "FAILED_RATE_LIMIT"
        return content.strip()
    except requests.ConnectionError:
        return "FAILED: NAT is not running. Run: bash gaia_tools/start_services.sh"
    except requests.Timeout:
        return "FAILED_TIMEOUT"
    except requests.HTTPError as e:
        err_text = str(e).lower()
        if "429" in err_text or any(s in err_text for s in _RATE_LIMIT_STRINGS):
            return "FAILED_RATE_LIMIT"
        return f"FAILED: {e}"
    except (KeyError, IndexError) as e:
        return f"FAILED: Unexpected NAT response format: {e}"
    except Exception as e:
        return f"FAILED: {e}"


def _extract_answer_from_think(inside: str) -> str | None:
    """Try to extract a clear answer from inside a <think> block.

    Handles models that state "answer: X" or "Thus answer: X" repeatedly
    inside their reasoning before the block closes or gets truncated.
    """
    answer_patterns = [
        r"(?:Thus |So |Therefore |The )?(?:final )?(?:answer|result)(?:\s+(?:is|was|should be|would be))?[:\s]+[\"']?([^\n\"']{1,200})[\"']?\s*$",
        r"(?:Thus |So )?(?:we should )?(?:output|respond with)[:\s]+[\"']?([^\n\"']{1,200})[\"']?\s*$",
        r"^([^\n]{1,100})$",
    ]
    lines = [ln.strip() for ln in inside.split("\n") if ln.strip()]
    if not lines:
        return None

    for line in reversed(lines):
        for pat in answer_patterns[:2]:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                # Truncate at sentence boundary: "Claus. However..." -> "Claus"
                sent_break = re.search(r"\.\s+[A-Z]", candidate)
                if sent_break:
                    candidate = candidate[:sent_break.start()]
                candidate = candidate.strip().rstrip(".")
                if len(candidate) > 1 and not candidate.lower().startswith(("but ", "however ", "let ", "we should", "should be", "would be", "could be", "we need")):
                    return candidate

    return lines[-1] if lines else None


def clean_answer(raw: str) -> str:
    """Post-process an agent answer to strip common formatting wrappers.

    GAIA uses exact-match scoring. Agents often return correct answers wrapped
    in natural language ("The answer is 555", "Answer: 555", etc.). This
    function strips those wrappers to recover points that would otherwise be lost.
    """
    text = raw.strip()
    if not text:
        return text

    # Strip <think>...</think> blocks FIRST so that FINAL ANSWER: patterns
    # rehearsed inside reasoning are not captured by the regex below.
    text_clean = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text_clean = re.sub(r"<think>.*", "", text_clean, flags=re.DOTALL).strip()

    # FINAL ANSWER: extraction (highest priority, used by top GAIA agents).
    fa_matches = list(re.finditer(r"FINAL ANSWER:\s*(.+)", text_clean, re.IGNORECASE))
    if fa_matches:
        text = fa_matches[-1].group(1).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
    elif text_clean:
        text = text_clean
    else:
        # Everything was inside <think>. Try to extract from think content.
        think_match = re.search(r"</think>\s*", text, re.DOTALL)
        if think_match:
            inside = text[len("<think>"):think_match.start()].strip()
            if not inside:
                return ""
            extracted = _extract_answer_from_think(inside)
            text = extracted if extracted else text
        elif text.startswith("<think>"):
            inside = text[len("<think>"):].strip()
            if not inside:
                return ""
            extracted = _extract_answer_from_think(inside)
            text = extracted if extracted else text

    prefixes = [
        "so the final answer is",
        "so the answer is",
        "so the answer should be",
        "so the answer was",
        "so the result is",
        "therefore the answer is",
        "thus the answer is",
        "the final answer is",
        "the answer is",
        "the answer should be",
        "the answer was",
        "the result is",
        "final answer:",
        "final answer is",
        "answer:",
        "answer is",
        "result:",
        "result is",
    ]
    changed = True
    while changed:
        changed = False
        text = text.lstrip(":").lstrip("-").strip()
        lower = text.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                text = text[len(prefix):].strip()
                lower = text.lower()
                changed = True
                break

    text = text.strip('"').strip("'").strip("`")

    # Strip LaTeX \boxed{...} wrapper
    boxed = re.match(r"^\\boxed\{(.+)\}$", text)
    if boxed:
        text = boxed.group(1).strip()

    if text.endswith(".") and not re.search(r"\d\.\d", text[-4:]):
        text = text[:-1].strip()

    if text.endswith(","):
        text = text[:-1].strip()

    if text.startswith("$"):
        text = text[1:].strip()

    # Strip commas from standalone numbers up to 999,999,999.
    # fullmatch ensures we only touch answers that are purely a single number,
    # not comma-separated lists like "132,133,134" or mixed text.
    stripped = text.strip()
    num_match = re.fullmatch(r'(-?)(\d{1,3}(?:,\d{3}){1,2})(\.\d+)?', stripped)
    if num_match:
        text = (num_match.group(1)
                + num_match.group(2).replace(',', '')
                + (num_match.group(3) or ''))
    elif "," in text:
        text = re.sub(r",(?=\S)", ", ", text)

    return text.strip()


def normalize_for_comparison(text: str) -> str:
    """Normalize text for GAIA-style comparison.

    Conservative normalizer: commas become spaces (preserving word boundaries),
    articles are stripped, and whitespace is collapsed.
    """
    s = text.strip().lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[,;!?()]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def check_answer(submitted: str, expected: str) -> bool:
    """Check if submitted answer matches expected using GAIA-style matching."""
    sub = normalize_for_comparison(submitted)
    exp = normalize_for_comparison(expected)
    if sub == exp:
        return True
    if exp in sub and len(exp) > 2:
        return True
    cleaned_sub = normalize_for_comparison(clean_answer(submitted))
    if cleaned_sub == exp:
        return True
    return False


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

GAIA_HF_REPO = "gaia-benchmark/GAIA"
GAIA_HF_SUBDIR = "2023/validation"


def download_gaia_files(questions: list[dict], files_dir: str) -> None:
    """Download attached files from the gated GAIA dataset on HuggingFace Hub.

    The HF scoring API's /files/{task_id} endpoint has been broken since Dec 2025,
    so we pull files directly from the source dataset instead.
    Requires: HF_TOKEN env var or `huggingface-cli login`, plus accepted dataset terms
    at https://huggingface.co/datasets/gaia-benchmark/GAIA

    If all files are already present in files_dir, no HF token is needed.
    """
    os.makedirs(files_dir, exist_ok=True)
    needed = [q.get("file_name", "") for q in questions if q.get("file_name")]
    if not needed:
        return

    already = sum(1 for fn in needed if os.path.exists(os.path.join(files_dir, fn)))
    missing = [fn for fn in needed if not os.path.exists(os.path.join(files_dir, fn))]
    print(f"GAIA files: {already} cached, {len(missing)} to download.")

    if not missing:
        return

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token() or ""
        except Exception:
            pass

    if not token:
        print("  WARN: No HF token found. Set HF_TOKEN or run `huggingface-cli login`.")
        print("  Skipping file download. Questions with attachments may fail.")
        return

    for fn in missing:
        dest = os.path.join(files_dir, fn)
        try:
            downloaded = hf_hub_download(
                repo_id=GAIA_HF_REPO,
                filename=f"{GAIA_HF_SUBDIR}/{fn}",
                repo_type="dataset",
                token=token,
            )
            shutil.copy2(downloaded, dest)
            print(f"  Downloaded {fn}")
        except Exception as e:
            print(f"  WARN: Could not download {fn}: {e}")


_YT_URL_RE = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})")


def build_question_prompt(question: dict, files_dir: str | None = None) -> str:
    """Build the full prompt for a GAIA question, routing to the correct tool."""
    text = question.get("question", question.get("input", ""))
    file_name = question.get("file_name", "")

    if file_name and files_dir:
        file_path = os.path.join(files_dir, file_name)
        if os.path.exists(file_path):
            ext = Path(file_name).suffix.lower()
            if ext in IMAGE_EXTS:
                tool_hint = (
                    f"Use the describe_image tool with path '{file_path}' "
                    "and a specific question about what you see."
                )
            elif ext in AUDIO_EXTS:
                tool_hint = (
                    f"Use the transcribe_audio tool with path '{file_path}' "
                    "to get the spoken content."
                )
            else:
                tool_hint = (
                    f"Use the read_file tool with path '{file_path}' "
                    "to access the file contents."
                )
            text += f"\n\n[Attached file available at: {file_path}]\n{tool_hint}"
        else:
            text += f"\n\n[Note: Referenced file '{file_name}' not found at {file_path}]"

    # YouTube URL detection: inject per-question hint with extracted video ID
    yt_match = _YT_URL_RE.search(text)
    if yt_match:
        vid = yt_match.group(1)
        text += (
            f"\n\n[IMPORTANT: This question references a YouTube video. "
            f"Call get_youtube_transcript with video_id='{vid}' FIRST "
            f"to get the spoken content before trying any web search.]"
        )

    return text


RETRY_PROMPT_TEMPLATE = (
    "Answer this question with ONLY the answer, no explanation.\n"
    "End with: FINAL ANSWER: [your answer]\n\n{question}"
)

_GARBAGE_STARTS = [
    "we need to", "let's ", "let us ", "we should",
    "the question", "the page", "the file", "the text",
    "let me ", "i need to", "i should", "i'll ", "i will ",
    "could you please", "i apologize", "i'm sorry", "i cannot",
    "the tool calling agent could not",
    "according to", "based on", "from the search",
    "the search results", "after searching", "upon review",
    "wait ", "wait,", "wait-", "actually ", "actually,",
    "hmm", "ok so ", "ok, so ",
    "use python", "now i need", "now we need", "first,",
    "looking at", "i'll focus", "i'm going",
]

_REASONING_PHRASES = [
    "let's search", "we need to find", "use python to",
    "let me search", "i'll need to", "we should look",
    "let's try", "let me try", "i need to find",
    "let's look at", "let's check",
]

_AGENT_FAILURE_PREFIXES = [
    "The tool calling agent could not",
    "The react agent could not",
    "The reasoning agent could not",
    "The agent could not",
    "Agent execution failed",
    "Router agent failed",
    "Router Agent failed",
]


def _looks_like_garbage(answer: str) -> bool:
    """Detect answers that are reasoning/prose instead of a real answer.

    Conservative: only flags answers that clearly look like reasoning,
    not legitimate answers that happen to be long (e.g. comma-separated lists).
    """
    if len(answer) > 300:
        return True
    lower = answer.lower()
    if any(lower.startswith(phrase) for phrase in _GARBAGE_STARTS):
        return True
    if any(phrase in lower for phrase in _REASONING_PHRASES):
        return True
    if len(answer) > 150 and any(w in lower for w in ("however", "therefore", "because")):
        return True
    return False


def _wait_for_rate_limit(attempt: int) -> float:
    """Exponential backoff for rate limits.

    NVIDIA Build free tier: ~40 RPM, 1,000-5,000 total credits.
    First retry at 5s covers a temporary burst. Second at 30s covers a full
    cooldown window. Third at 60s is a final attempt; if this also 429s,
    credits are likely exhausted (no amount of waiting helps).
    """
    delays = [5, 30, 60]
    wait = delays[min(attempt, len(delays) - 1)]
    print(f"    [rate limited] Waiting {wait}s before retry (attempt {attempt + 1}/3)...")
    time.sleep(wait)
    return wait


def ask_with_retry(question_text: str, timeout: int) -> tuple[str, float, bool]:
    """Ask NAT, retry on timeout or garbage answer. Returns (answer, elapsed, was_retry)."""
    start = time.time()
    answer = ask_nat(question_text, timeout=timeout)
    elapsed = time.time() - start

    if answer == "FAILED_RATE_LIMIT":
        for attempt in range(3):
            _wait_for_rate_limit(attempt)
            answer = ask_nat(question_text, timeout=timeout)
            elapsed = time.time() - start
            if answer != "FAILED_RATE_LIMIT":
                break
        if answer == "FAILED_RATE_LIMIT":
            elapsed = time.time() - start
            return (
                "FAILED: NVIDIA Build returned 429 after 3 retries (~95s of waiting). "
                "This usually means your NVIDIA Build credits are exhausted. "
                "Check your balance at https://build.nvidia.com/settings/api-keys "
                "or switch to a local agent."
            ), elapsed, True

    # NAT structural failure (hit max_iterations): retry with a fresh, directive
    # prompt. The original attempt accumulated a long context that may have caused
    # the agent to loop. A clean conversation often takes a different search path.
    if any(answer.startswith(pfx) for pfx in _AGENT_FAILURE_PREFIXES):
        needs_retry = True
        retry_reason = "max_iterations"
    else:
        needs_retry = False
        retry_reason = ""

    if not needs_retry and answer == "FAILED_TIMEOUT":
        needs_retry = True
        retry_reason = "timeout"

    # Transient HTTP failures (422 context exhaustion, 500/502/503 server errors).
    # A fresh conversation via RETRY_PROMPT_TEMPLATE bypasses the original failure.
    if not needs_retry and answer.startswith("FAILED:"):
        needs_retry = True
        retry_reason = "http_error"

    # Truncated reasoning: model hit max_tokens inside <think> without concluding.
    if not needs_retry and answer.startswith("<think>") and "</think>" not in answer:
        needs_retry = True
        retry_reason = "truncated_think"

    if not needs_retry and not answer.startswith("FAILED:"):
        cleaned_check = clean_answer(answer)
        if _looks_like_garbage(cleaned_check):
            needs_retry = True
            retry_reason = "garbage"

    if needs_retry:
        print(f"    [retry reason: {retry_reason}]")
        # Preserve tool hints appended at end (file paths, YouTube IDs)
        hint_start = question_text.find("\n\n[")
        if hint_start > 0:
            body = question_text[:hint_start][:1500]
            hints = question_text[hint_start:][:500]
            retry_q = body + hints
        else:
            retry_q = question_text[:2000]
        short_prompt = RETRY_PROMPT_TEMPLATE.format(question=retry_q)
        retry_timeout = min(timeout, 180) if retry_reason == "timeout" else timeout
        retry_start = time.time()
        retry_answer = ask_nat(short_prompt, timeout=retry_timeout)
        elapsed += time.time() - retry_start

        if retry_answer == "FAILED_RATE_LIMIT":
            _wait_for_rate_limit(0)
            retry_answer = ask_nat(short_prompt, timeout=retry_timeout)
            elapsed = time.time() - start

        if retry_answer == "FAILED_TIMEOUT":
            if answer != "FAILED_TIMEOUT":
                return answer, elapsed, True
            answer = f"FAILED: Timed out after {timeout}s (including retry)"
        elif retry_answer.startswith("FAILED:"):
            pass
        elif retry_reason == "garbage":
            cleaned_retry = clean_answer(retry_answer)
            if _looks_like_garbage(cleaned_retry):
                cleaned_orig = clean_answer(answer)
                if len(cleaned_retry) < len(cleaned_orig):
                    answer = retry_answer
            else:
                answer = retry_answer
        else:
            answer = retry_answer
        return answer, elapsed, True

    return answer, elapsed, False


_VLLM_HEALTH = "http://localhost:9000/health"
_IS_CLOUD = os.environ.get("USES_LOCAL_LLM", "yes") == "no"


def _wait_for_vllm(max_wait: int = 30) -> None:
    """Block until vLLM responds on /health. Skipped for non-local agents."""
    if _IS_CLOUD:
        return
    for i in range(max_wait):
        try:
            r = requests.get(_VLLM_HEALTH, timeout=3)
            if r.status_code < 500:
                if i > 0:
                    print(f"    [vLLM recovered after {i}s]")
                return
        except Exception:
            pass
        time.sleep(1)
    print(f"    [WARNING: vLLM unresponsive after {max_wait}s]")


def print_level_report(results: list[dict]) -> dict:
    """Print per-level scores and weighted total. Returns summary dict."""
    by_level: dict[int, list[dict]] = {}
    for r in results:
        level = r.get("level", 1)
        by_level.setdefault(level, []).append(r)

    has_scoring = any(r.get("expected_answer") for r in results)

    weighted_total = 0
    max_weighted = 0
    level_summaries = {}
    total_time = sum(r.get("elapsed_seconds", 0) for r in results)

    for level in sorted(by_level.keys()):
        items = by_level[level]
        failures = [r for r in items if r.get("failed")]
        total = len(items)
        answered = total - len(failures)
        time_avg = sum(r.get("elapsed_seconds", 0) for r in items) / max(total, 1)

        if has_scoring:
            correct = sum(1 for r in items if r.get("correct", False))
            pct = 100 * correct / total if total > 0 else 0
            weight = LEVEL_WEIGHTS.get(level, 1)
            weighted = correct * weight
            max_w = total * weight
            weighted_total += weighted
            max_weighted += max_w
            level_summaries[level] = {
                "correct": correct, "total": total, "pct": pct,
                "weighted": weighted, "weight": weight,
            }
        else:
            level_summaries[level] = {
                "answered": answered, "total": total, "failures": len(failures),
            }

    if has_scoring:
        overall_correct = sum(s["correct"] for s in level_summaries.values())
        overall_total = sum(s["total"] for s in level_summaries.values())
        overall_pct = 100 * overall_correct / overall_total if overall_total > 0 else 0

        print("\n" + "=" * 60)
        print("GAIA BENCHMARK RESULTS")
        print("=" * 60)
        for level in sorted(by_level.keys()):
            items = by_level[level]
            failures = [r for r in items if r.get("failed")]
            s = level_summaries[level]
            time_avg = sum(r.get("elapsed_seconds", 0) for r in items) / max(len(items), 1)
            weight = LEVEL_WEIGHTS.get(level, 1)
            print(f"\n  Level {level}: {s['correct']}/{s['total']} ({s['pct']:.1f}%)  "
                  f"x{weight} = {s['weighted']} pts  "
                  f"[avg {time_avg:.1f}s/q, {len(failures)} failed]")
        print(f"\n  {'─' * 50}")
        print(f"  Raw score:      {overall_correct}/{overall_total} ({overall_pct:.1f}%)")
        print(f"  Weighted total: {weighted_total}/{max_weighted}")
        print(f"  Total time:     {total_time:.0f}s ({total_time/60:.1f} min)")
        print("=" * 60)
    else:
        overall_answered = sum(s["answered"] for s in level_summaries.values())
        overall_total = sum(s["total"] for s in level_summaries.values())
        overall_failures = sum(s["failures"] for s in level_summaries.values())
        fail_note = f" ({overall_failures} failed)" if overall_failures else ""
        print(f"\n  {overall_answered}/{overall_total} questions completed{fail_note}"
              f" in {total_time:.0f}s ({total_time/60:.1f} min)")

    summary = {
        "has_scoring": has_scoring,
        "total_time_seconds": round(total_time, 1),
        "levels": level_summaries,
    }
    if has_scoring:
        summary["raw_correct"] = overall_correct
        summary["raw_total"] = overall_total
        summary["raw_pct"] = round(overall_pct, 1)
        summary["weighted_total"] = weighted_total
        summary["weighted_max"] = max_weighted
    else:
        summary["answered"] = overall_answered
        summary["total"] = overall_total
        summary["failures"] = overall_failures

    return summary


def main():
    parser = argparse.ArgumentParser(description="GAIA benchmark runner")
    parser.add_argument("username", nargs="?", help="Team name (e.g., NAT-UCB-AgentSmiths, NAT-Stanford-TeamX)")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], help="Run only this level")
    parser.add_argument("--local", action="store_true", help="Use local dataset, skip HF submit")
    parser.add_argument("--dataset", default="gaia_questions.json", help="Local dataset path")
    parser.add_argument("--files-dir", default="./gaia_files", help="Directory with attached files")
    parser.add_argument("--timeout", type=int, default=300, help="Per-question timeout (seconds)")
    parser.add_argument("--limit", type=int, help="Limit number of questions (for quick testing)")
    parser.add_argument("--no-clean", action="store_true",
                        help="Disable answer post-processing (submit raw agent output)")
    submit_group = parser.add_mutually_exclusive_group()
    submit_group.add_argument("--submit", action="store_true",
                              help="Auto-submit to HF leaderboard (nohup-safe, no prompts)")
    submit_group.add_argument("--no-submit", action="store_true",
                              help="Skip HF leaderboard submission entirely")
    args = parser.parse_args()

    if not args.username:
        if args.submit:
            print("Error: --submit requires a username argument.")
            sys.exit(1)
        args.username = input("Team name (e.g., NAT-UCB-AgentSmiths, NAT-Stanford-TeamX): ")

    questions: list[dict] = []
    use_local = args.local

    if not use_local:
        try:
            print("Fetching GAIA questions from HuggingFace API...")
            resp = requests.get(f"{API_URL}/questions", timeout=15)
            resp.raise_for_status()
            questions = resp.json()
            print(f"Got {len(questions)} questions from HuggingFace.\n")
        except Exception as e:
            print(f"HF API unavailable ({e}). Falling back to local dataset.")
            use_local = True

    if use_local:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: {dataset_path} not found. Download GAIA validation set first.")
            sys.exit(1)
        questions = json.loads(dataset_path.read_text())
        print(f"Loaded {len(questions)} questions from {dataset_path}.\n")

    if args.level:
        questions = [q for q in questions if int(q.get("level", q.get("Level", 1))) == args.level]
        print(f"Filtered to {len(questions)} Level {args.level} questions.\n")

    if args.limit:
        questions = questions[:args.limit]
        print(f"Limited to first {args.limit} questions.\n")

    if not questions:
        print("No questions to process. Exiting.")
        sys.exit(1)

    print("Checking for attached files to download...")
    download_gaia_files(questions, args.files_dir)
    print()

    results: list[dict] = []
    answers_for_hf: list[dict] = []
    benchmark_start = time.time()
    prev_was_failed = False

    for i, q in enumerate(questions):
        if prev_was_failed:
            _wait_for_vllm()

        task_id = q.get("task_id", f"local_{i}")
        level = int(q.get("level", q.get("Level", 1)))
        question_text = build_question_prompt(q, args.files_dir)
        expected = q.get("Final answer", q.get("expected_answer", q.get("answer", "")))

        short_q = q.get("question", q.get("input", ""))[:80]
        print(f"[{i+1}/{len(questions)}] (L{level}) {short_q}...")

        raw_answer, elapsed, was_retry = ask_with_retry(question_text, args.timeout)
        is_failed = raw_answer.startswith("FAILED:")

        submitted = raw_answer
        if not is_failed and not args.no_clean:
            submitted = clean_answer(raw_answer)

        is_correct = False
        has_expected = bool(expected)
        if has_expected and not is_failed:
            is_correct = check_answer(submitted, str(expected))

        if is_failed:
            status = "FAILED"
        elif is_correct:
            status = "CORRECT"
        elif has_expected:
            status = "WRONG"
        else:
            status = "SUBMITTED"

        retry_tag = " [RETRY]" if was_retry else ""
        cleaned_tag = ""
        if submitted != raw_answer and not is_failed:
            cleaned_tag = f" (cleaned: {submitted[:60]})"

        print(f"  [{status}]{retry_tag} ({elapsed:.1f}s) -> {raw_answer[:100]}{cleaned_tag}")
        if not is_correct and has_expected and not is_failed:
            print(f"  Expected: {expected}")
        print()

        results.append({
            "task_id": task_id,
            "level": level,
            "question": q.get("question", q.get("input", ""))[:200],
            "raw_answer": raw_answer,
            "submitted_answer": submitted,
            "expected_answer": str(expected) if expected else None,
            "correct": is_correct,
            "failed": is_failed,
            "was_retry": was_retry,
            "elapsed_seconds": round(elapsed, 1),
        })
        answers_for_hf.append({"task_id": task_id, "submitted_answer": submitted})
        prev_was_failed = is_failed

    total_elapsed = time.time() - benchmark_start

    LOCAL_RESULTS.write_text(json.dumps(results, indent=2))

    summary = print_level_report(results)

    should_submit = False
    if not use_local and not args.no_submit:
        if args.submit:
            should_submit = True
        else:
            print()
            try:
                choice = input("Submit to PUBLIC HuggingFace leaderboard? [y/N]: ").strip().lower()
                should_submit = choice in ("y", "yes")
            except (OSError, EOFError):
                print("Non-interactive mode detected. Use --submit to auto-submit.")

    if should_submit:
        print("\nSubmitting to HuggingFace leaderboard...", end="", flush=True)
        try:
            result = requests.post(f"{API_URL}/submit", json={
                "username": args.username,
                "agent_code": f"https://huggingface.co/{args.username}",
                "answers": answers_for_hf,
            }, timeout=60)
            result.raise_for_status()
            data = result.json()
            hf_score = data.get("score", "N/A")
            hf_correct = data.get("correct_count", "?")
            hf_total = data.get("total_attempted", "?")
            hf_message = data.get("message", "")
            print(f" done.\n")
            print(f"  HF Score:   {hf_correct}/{hf_total} correct ({hf_score}%)")
            if hf_message:
                print(f"  HF Message: {hf_message}")
            summary["hf_score"] = hf_score
            summary["hf_correct"] = hf_correct
            summary["hf_total"] = hf_total
            summary["hf_message"] = hf_message
        except Exception as e:
            print(f" failed.\n  {e}\n  Results saved locally in {LOCAL_RESULTS}.")
    elif not use_local and not args.no_submit:
        print("Skipped public submission. Results saved locally only.")

    # Print quick diagnostic: failures and slowest questions
    failed_results = [r for r in results if r.get("failed")]
    if failed_results:
        print(f"\n  Failed to answer ({len(failed_results)}):")
        for r in failed_results:
            print(f"    Q{results.index(r)+1}: {r['question'][:70]}...")
    slow = sorted(results, key=lambda r: r.get("elapsed_seconds", 0), reverse=True)[:3]
    if slow and slow[0].get("elapsed_seconds", 0) > 60:
        print(f"\n  Slowest questions:")
        for r in slow:
            if r.get("elapsed_seconds", 0) > 60:
                retry = " [retry]" if r.get("was_retry") else ""
                print(f"    Q{results.index(r)+1}: {r['elapsed_seconds']:.0f}s{retry}"
                      f" - {r['question'][:60]}...")

    summary_path = Path("gaia_summary.json")
    summary["username"] = args.username
    summary["wall_clock_seconds"] = round(total_elapsed, 1)
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
