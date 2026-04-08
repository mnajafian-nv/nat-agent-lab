"""Tests for ask.py pure functions.

Covers config detection, question handling, YAML validation, and level indexing
— all the logic that changed during the cloud/nogpu removal and dev/test split.
"""
import os
import json
import tempfile
import textwrap

import pytest

from ask import (
    _uses_local_llm,
    _extract_final_answer,
    _build_level_index,
    _resolve_gaia,
    _validate_yaml,
    build_question_prompt,
    normalize_for_comparison,
    load_gaia_questions,
    load_gaia_dev_questions,
    parse_agent_info,
    AGENTS,
)


# ---------------------------------------------------------------------------
# _uses_local_llm
# ---------------------------------------------------------------------------

class TestUsesLocalLlm:
    """Detect whether a config needs local vLLM."""

    def _write_yaml(self, tmp_path, content):
        p = tmp_path / "agent.yml"
        p.write_text(textwrap.dedent(content))
        return str(p)

    def test_local_vllm(self, tmp_path):
        path = self._write_yaml(tmp_path, """\
            llms:
              llm:
                _type: openai
                base_url: "http://localhost:9000/v1"
            workflow:
              llm_name: llm
        """)
        assert _uses_local_llm(path) is True

    def test_ollama_not_local_vllm(self, tmp_path):
        path = self._write_yaml(tmp_path, """\
            llms:
              llm:
                _type: openai
                base_url: "http://localhost:11434/v1"
            workflow:
              llm_name: llm
        """)
        assert _uses_local_llm(path) is False

    def test_nim_not_local(self, tmp_path):
        path = self._write_yaml(tmp_path, """\
            llms:
              llm:
                _type: nim
                model_name: some-model
            workflow:
              llm_name: llm
        """)
        assert _uses_local_llm(path) is False

    def test_remote_url_not_local(self, tmp_path):
        path = self._write_yaml(tmp_path, """\
            llms:
              llm:
                _type: openai
                base_url: "https://api.example.com/v1"
            workflow:
              llm_name: llm
        """)
        assert _uses_local_llm(path) is False

    def test_missing_file_defaults_true(self):
        assert _uses_local_llm("/nonexistent/path.yml") is True

    def test_no_base_url_defaults_true(self, tmp_path):
        path = self._write_yaml(tmp_path, """\
            llms:
              llm:
                _type: openai
            workflow:
              llm_name: llm
        """)
        assert _uses_local_llm(path) is True


# ---------------------------------------------------------------------------
# _extract_final_answer
# ---------------------------------------------------------------------------

class TestExtractFinalAnswer:
    def test_basic(self):
        assert _extract_final_answer("FINAL ANSWER: Paris") == "Paris"

    def test_case_insensitive(self):
        assert _extract_final_answer("final answer: 42") == "42"

    def test_strips_think_block(self):
        assert _extract_final_answer(
            "<think>reasoning</think>\nFINAL ANSWER: yes"
        ) == "yes"

    def test_unclosed_think_block(self):
        result = _extract_final_answer("<think>stuck forever")
        assert result == ""

    def test_no_marker_returns_full_text(self):
        assert _extract_final_answer("just a plain answer") == "just a plain answer"

    def test_last_marker_wins(self):
        assert _extract_final_answer(
            "FINAL ANSWER: wrong\nFINAL ANSWER: right"
        ) == "right"


# ---------------------------------------------------------------------------
# _build_level_index / _resolve_gaia
# ---------------------------------------------------------------------------

_SAMPLE_QUESTIONS = [
    {"question": "Q1", "level": "1"},
    {"question": "Q2", "level": "1"},
    {"question": "Q3", "level": "2"},
    {"question": "Q4", "level": "3"},
]


class TestBuildLevelIndex:
    def test_groups_by_level(self):
        idx = _build_level_index(_SAMPLE_QUESTIONS)
        assert len(idx["1"]) == 2
        assert len(idx["2"]) == 1
        assert len(idx["3"]) == 1

    def test_flat_idx_preserved(self):
        idx = _build_level_index(_SAMPLE_QUESTIONS)
        flat_idx, q = idx["2"][0]
        assert flat_idx == 2
        assert q["question"] == "Q3"

    def test_empty_list(self):
        assert _build_level_index([]) == {}


class TestResolveGaia:
    def test_valid(self):
        flat_idx, q = _resolve_gaia(_SAMPLE_QUESTIONS, "1", 2)
        assert q["question"] == "Q2"

    def test_invalid_level(self, capsys):
        flat_idx, q = _resolve_gaia(_SAMPLE_QUESTIONS, "9", 1)
        assert q is None

    def test_invalid_pos(self, capsys):
        flat_idx, q = _resolve_gaia(_SAMPLE_QUESTIONS, "1", 99)
        assert q is None


# ---------------------------------------------------------------------------
# _validate_yaml
# ---------------------------------------------------------------------------

class TestValidateYaml:
    def _write(self, tmp_path, content, name="agent.yml"):
        p = tmp_path / name
        p.write_text(textwrap.dedent(content))
        return str(p)

    def test_valid_config(self, tmp_path):
        path = self._write(tmp_path, """\
            llms:
              my_llm:
                _type: openai
            workflow:
              _type: tool_calling_agent
              llm_name: my_llm
        """)
        ok, msg = _validate_yaml(path)
        assert ok is True

    def test_missing_file(self):
        ok, msg = _validate_yaml("/nonexistent.yml")
        assert ok is False
        assert "not found" in msg.lower()

    def test_no_workflow(self, tmp_path):
        path = self._write(tmp_path, "llms:\n  x:\n    _type: openai\n")
        ok, msg = _validate_yaml(path)
        assert ok is False
        assert "workflow" in msg.lower()

    def test_bad_workflow_type(self, tmp_path):
        path = self._write(tmp_path, """\
            llms:
              llm:
                _type: openai
            workflow:
              _type: invalid_agent
              llm_name: llm
        """)
        ok, msg = _validate_yaml(path)
        assert ok is False
        assert "invalid_agent" in msg

    def test_missing_llm_name(self, tmp_path):
        path = self._write(tmp_path, """\
            llms:
              llm:
                _type: openai
            workflow:
              _type: tool_calling_agent
        """)
        ok, msg = _validate_yaml(path)
        assert ok is False
        assert "llm_name" in msg.lower()

    def test_llm_not_in_llms_section(self, tmp_path):
        path = self._write(tmp_path, """\
            llms:
              other_llm:
                _type: openai
            workflow:
              _type: tool_calling_agent
              llm_name: missing_llm
        """)
        ok, msg = _validate_yaml(path)
        assert ok is False
        assert "missing_llm" in msg

    def test_empty_path(self):
        ok, msg = _validate_yaml("")
        assert ok is False

    def test_invalid_yaml(self, tmp_path):
        path = self._write(tmp_path, "{{not valid yaml")
        ok, msg = _validate_yaml(path)
        assert ok is False


# ---------------------------------------------------------------------------
# AGENTS dict
# ---------------------------------------------------------------------------

class TestAgentsDict:
    """Verify the agent registry matches expectations after nogpu removal."""

    def test_expected_agents(self):
        assert set(AGENTS.keys()) == {"single", "multi", "ultrafast", "ollama"}

    def test_no_nogpu(self):
        assert "ultrafast-nogpu" not in AGENTS

    def test_all_configs_are_yaml(self):
        for name, path in AGENTS.items():
            assert path.endswith(".yml"), f"{name} config should be a .yml file"


# ---------------------------------------------------------------------------
# parse_agent_info
# ---------------------------------------------------------------------------

class TestParseAgentInfo:
    def test_parses_ultrafast(self):
        path = AGENTS.get("ultrafast")
        if path and os.path.exists(path):
            info = parse_agent_info(path)
            assert info["type"] == "tool_calling_agent"
            assert "error" not in info

    def test_parses_ollama(self):
        path = AGENTS.get("ollama")
        if path and os.path.exists(path):
            info = parse_agent_info(path)
            assert info["type"] == "tool_calling_agent"
            assert "error" not in info
            assert "describe_image" in info["tools"]

    def test_missing_file(self):
        info = parse_agent_info("/nonexistent.yml")
        assert "error" in info


# ---------------------------------------------------------------------------
# build_question_prompt
# ---------------------------------------------------------------------------

class TestBuildQuestionPrompt:
    def test_plain_question(self):
        q = {"question": "What is 2+2?"}
        prompt = build_question_prompt(q)
        assert "2+2" in prompt

    def test_question_with_file(self, tmp_path):
        # Create a fake file so the prompt includes the file hint
        fake_file = tmp_path / "data.csv"
        fake_file.write_text("a,b\n1,2\n")
        q = {"question": "Analyze this data", "file_name": "data.csv"}
        # build_question_prompt uses GAIA_FILES_DIR; test the basic case
        prompt = build_question_prompt(q)
        assert "Analyze this data" in prompt


# ---------------------------------------------------------------------------
# load_gaia_questions / load_gaia_dev_questions
# ---------------------------------------------------------------------------

class TestLoadQuestions:
    def test_load_returns_list(self):
        questions = load_gaia_questions()
        # May be empty if file doesn't exist, but should be a list
        assert isinstance(questions, list)

    def test_load_dev_returns_list(self):
        dev = load_gaia_dev_questions()
        assert isinstance(dev, list)

    def test_questions_have_no_answers(self):
        """Test questions (gaia_questions.json) should not have Final answer."""
        questions = load_gaia_questions()
        if questions:
            for q in questions:
                assert "Final answer" not in q, \
                    "Test questions should not contain answers"

    def test_dev_questions_have_answers(self):
        """Dev questions should have Final answer field."""
        dev = load_gaia_dev_questions()
        if dev:
            with_answers = sum(1 for q in dev if q.get("Final answer"))
            assert with_answers == len(dev), \
                "All dev questions should have answers"
