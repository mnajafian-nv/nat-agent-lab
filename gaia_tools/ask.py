#!/usr/bin/env python3
"""Interactive chat interface for exploring NAT agents on GAIA questions.

Usage:
    ./ask
"""

import json
import os
import re
import readline
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import requests
    import yaml
except ImportError:
    print("\n  Run ./ask instead. It sets up the environment automatically.\n")
    sys.exit(1)

_gaia_check = None
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gaia_submit import check_answer as _gaia_check
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


NAT_URL = "http://localhost:8000/v1/chat/completions"
VLLM_HEALTH = "http://localhost:9000/health"
NAT_HEALTH = "http://localhost:8000/health"
PHOENIX_HEALTH = "http://localhost:6006"

AGENTS = {
    "single": "single-agent/gaia_agent.yml",
    "multi": "multi-agent/gaia_agent_multi.yml",
    "ultrafast": "ultrafast-agent/gaia_agent_ultrafast.yml",
    "ultrafast-nogpu": "ultrafast-nogpu-agent/gaia_agent_ultrafast_nogpu.yml",
}

HISTORY_FILE = Path.home() / ".nat_ask_history"

GAIA_QUESTIONS_FILE = ROOT / "gaia_questions.json"
GAIA_FILES_DIR = ROOT / "gaia_files"

TAVILY_TEST_URL = "https://api.tavily.com/search"
NGC_TEST_URL = "https://integrate.api.nvidia.com/v1/models"
HF_TEST_URL = "https://huggingface.co/api/whoami-v2"

KEY_DISPLAY = {
    "TAVILY_API_KEY": "TAVILY_API_KEY (search)",
    "NGC_API_KEY": "NGC_API_KEY (NVIDIA Build models)",
    "HF_TOKEN": "HF_TOKEN (GAIA + HuggingFace)",
}

KEY_PREFIXES = {
    "TAVILY_API_KEY": "tvly-",
    "NGC_API_KEY": "nvapi-",
    "HF_TOKEN": "hf_",
}

IS_REMOTE = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))


def _normalize_env():
    """Self-heal .env: strip 'export', deduplicate keys, normalize quoting.

    Runs once at startup so that no matter how the file was created (setup.sh,
    manual paste, old code), it ends up in a clean KEY='value' format.  Idempotent.
    """
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    try:
        raw = env_file.read_text().splitlines()
    except Exception:
        return

    seen = {}   # key -> clean line (preserves last-wins semantics)
    order = []  # first-appearance order of keys
    others = [] # comments and blank lines (kept at top)

    for line in raw:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            others.append(line)
            continue
        bare = stripped[7:] if stripped.startswith("export ") else stripped
        if "=" not in bare:
            others.append(line)
            continue
        k, v = bare.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'\"")
        if k not in seen:
            order.append(k)
        seen[k] = v

    clean = others[:]
    for k in order:
        v = seen[k]
        clean.append(f"{k}='{v}'" if v else f"{k}=")

    new_text = "\n".join(clean) + "\n"
    try:
        old_text = env_file.read_text()
    except Exception:
        old_text = ""

    if new_text != old_text:
        try:
            env_file.write_text(new_text)
        except Exception:
            pass

    # Force-sync API keys from .env into os.environ so that keys updated
    # via ./ask or edited manually are always picked up.
    api_key_names = {"TAVILY_API_KEY", "NGC_API_KEY", "HF_TOKEN"}
    for k, v in seen.items():
        if v and (k in api_key_names or not os.environ.get(k)):
            os.environ[k] = v

    # Some NAT components (vision tools, etc.) look for NVIDIA_API_KEY.
    ngc = os.environ.get("NGC_API_KEY", "")
    if ngc and not os.environ.get("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = ngc


def _load_api_keys():
    """Load API keys from environment, falling back to .env file."""
    keys = {}
    for name in ("TAVILY_API_KEY", "NGC_API_KEY", "HF_TOKEN"):
        keys[name] = os.environ.get(name, "")
    env_file = ROOT / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k in keys and not keys[k]:
                        keys[k] = v
        except Exception:
            pass
    return keys


def _save_api_key(name, value):
    """Save a key to .env (update if exists, append if not). Also export to current process."""
    env_file = ROOT / ".env"
    safe_line = f"{name}='{value}'"

    if env_file.exists():
        try:
            raw = env_file.read_text()
        except Exception:
            try:
                with env_file.open("a") as f:
                    f.write(f"\n{safe_line}\n")
                os.environ[name] = value
                return True
            except Exception:
                return False

        lines = []
        found = False
        for line in raw.splitlines():
            stripped = line.strip()
            bare = stripped[7:] if stripped.startswith("export ") else stripped
            if bare.startswith(f"{name}="):
                if not found:
                    lines.append(safe_line)
                    found = True
            else:
                lines.append(line)

        if not found:
            lines.append(safe_line)

        try:
            env_file.write_text("\n".join(lines) + "\n")
            os.environ[name] = value
            return True
        except Exception:
            return False
    else:
        try:
            env_file.write_text(safe_line + "\n")
            os.environ[name] = value
            return True
        except Exception:
            return False


def _check_tavily(key):
    """Live-test a Tavily key. Returns a status string."""
    if not key:
        return "NOT SET"
    try:
        resp = requests.post(
            TAVILY_TEST_URL,
            json={"api_key": key, "query": "test", "max_results": 1},
            timeout=10,
        )
        if resp.status_code == 200:
            return "OK"
        if resp.status_code in (401, 403):
            return "INVALID KEY"
        if resp.status_code in (402, 429):
            return "OUT OF CREDITS"
        return f"ERROR ({resp.status_code})"
    except requests.Timeout:
        return "TIMEOUT"
    except Exception:
        return "UNREACHABLE"


def _check_ngc(key):
    """Live-test an NGC / NVIDIA Build key. Returns a status string."""
    if not key:
        return "NOT SET"
    try:
        resp = requests.get(
            NGC_TEST_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return "OK"
        if resp.status_code in (401, 403):
            return "INVALID KEY"
        if resp.status_code == 429:
            return "RATE LIMITED"
        return f"ERROR ({resp.status_code})"
    except requests.Timeout:
        return "TIMEOUT"
    except Exception:
        return "UNREACHABLE"


def _check_hf(token):
    """Live-test a HuggingFace token. Returns a status string."""
    if not token:
        return "NOT SET"
    try:
        resp = requests.get(
            HF_TEST_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return "OK"
        if resp.status_code in (401, 403):
            return "INVALID / EXPIRED"
        return f"ERROR ({resp.status_code})"
    except requests.Timeout:
        return "TIMEOUT"
    except Exception:
        return "UNREACHABLE"


def check_service(url, timeout=3):
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


MAX_TURNS = 20
CONFIG_CONTEXT_ROLE = "user"


def build_config_context(config_path):
    """Build a context message with the current agent config for grounded answers."""
    try:
        config_text = Path(config_path).read_text()
        return {
            "role": CONFIG_CONTEXT_ROLE,
            "content": (
                f"You are running as a NeMo Agent Toolkit (NAT) agent. "
                f"Your config file is '{config_path}'. Here is your full YAML config:\n\n"
                f"```yaml\n{config_text}\n```\n\n"
                f"When the user asks about your tools, config, or how to modify the YAML, "
                f"answer based on this actual config. Do not guess."
            ),
        }
    except Exception:
        return None


_RATE_LIMIT_STRINGS = ["rate limit", "rate_limit", "too many requests", "429", "quota exceeded"]


def ask_nat(messages, timeout=300):
    try:
        resp = requests.post(NAT_URL, json={
            "messages": messages,
        }, timeout=timeout)
        if resp.status_code == 429:
            return "FAILED: Rate limited by LLM provider. Wait a minute and try again, or type 'status' to check your API keys."
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            err_detail = str(data.get("error", data.get("detail", "")))
            if any(s in err_detail.lower() for s in _RATE_LIMIT_STRINGS):
                return "FAILED: Rate limited by LLM provider. Wait a minute and try again."
            return "FAILED: NAT returned empty response (no choices)."
        content = choices[0].get("message", {}).get("content")
        if content is None:
            return "FAILED: NAT returned null content."
        content_stripped = content.strip()
        if any(s in content_stripped.lower() for s in _RATE_LIMIT_STRINGS) and len(content_stripped) < 200:
            return "FAILED: Rate limited by LLM provider. Wait a minute and try again."
        return content_stripped
    except requests.ConnectionError:
        return "FAILED: NAT is not running. Use 'switch' to restart."
    except requests.Timeout:
        return "FAILED: Request timed out (300s). Try a simpler question."
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        detail = ""
        if e.response is not None:
            try:
                body = e.response.json()
                detail = body.get("detail", body.get("error", {}).get("message", ""))
                if isinstance(detail, dict):
                    detail = detail.get("message", str(detail))
            except Exception:
                detail = (e.response.text or "")[:300]
        detail = str(detail).strip()
        if status == 429:
            return "FAILED: Rate limited by LLM provider. Wait a minute and try again."
        if status == 422:
            base = f"FAILED: LLM request failed (422)"
            if detail:
                return f"{base}: {detail}"
            return f"{base}. Check credits: https://build.nvidia.com/settings/api-keys"
        if detail:
            return f"FAILED: HTTP {status}: {detail}"
        return f"FAILED: {e}"
    except Exception as e:
        return f"FAILED: {e}"


def load_gaia_questions():
    if not GAIA_QUESTIONS_FILE.exists():
        return []
    try:
        data = json.loads(GAIA_QUESTIONS_FILE.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: Failed to load {GAIA_QUESTIONS_FILE}: {e}")
        return []


def build_question_prompt(question):
    """Build prompt with file hints, matching gaia_submit.py logic."""
    text = question.get("question", question.get("input", ""))
    file_name = question.get("file_name", "")

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

    if file_name:
        file_path = str(GAIA_FILES_DIR / file_name)
        if os.path.exists(file_path):
            ext = Path(file_name).suffix.lower()
            if ext in IMAGE_EXTS:
                hint = (f"Use the describe_image tool with path '{file_path}' "
                        "and a specific question about what you see.")
            elif ext in AUDIO_EXTS:
                hint = (f"Use the transcribe_audio tool with path '{file_path}' "
                        "to get the spoken content.")
            else:
                hint = (f"Use the read_file tool with path '{file_path}' "
                        "to access the file contents.")
            text += f"\n\n[Attached file available at: {file_path}]\n{hint}"
        else:
            text += f"\n\n[Note: File '{file_name}' not found at {file_path}]"

    yt_match = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})", text)
    if yt_match:
        vid = yt_match.group(1)
        text += (f"\n\n[IMPORTANT: This question references a YouTube video. "
                 f"Call get_youtube_transcript with video_id='{vid}' FIRST "
                 f"to get the spoken content before trying any web search.]")

    return text


def _extract_final_answer(raw):
    """Extract the submitted answer from a full agent response.

    Strips <think> blocks and looks for 'FINAL ANSWER: X'. Falls back to
    the full response (minus think blocks) if no FINAL ANSWER marker found.
    """
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    fa = list(re.finditer(r"FINAL ANSWER:\s*(.+)", text, re.IGNORECASE))
    if fa:
        return fa[-1].group(1).strip()
    return text


def normalize_for_comparison(text):
    s = text.strip().lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[,;!?()]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def spinner_while(event, prefix="  Thinking"):
    chars = ["|", "/", "-", "\\"]
    i = 0
    start = time.time()
    while not event.is_set():
        elapsed = time.time() - start
        sys.stdout.write(f"\r{prefix} {chars[i % 4]} ({elapsed:.0f}s)")
        sys.stdout.flush()
        time.sleep(0.25)
        i += 1
    elapsed = time.time() - start
    sys.stdout.write(f"\r{prefix} done ({elapsed:.1f}s)    \n")
    sys.stdout.flush()


def ask_with_spinner(messages, timeout=300):
    done = threading.Event()
    result = [None]

    def worker():
        result[0] = ask_nat(messages, timeout=timeout)
        done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    spinner_while(done)
    t.join()
    return result[0]


_NOT_RECOVERABLE = ["credit", "quota", "billing", "unauthorized", "forbidden", "invalid api"]


def _should_auto_recover(answer, config_path):
    """True if the failure is recoverable by restarting NAT (cloud 422 only)."""
    if not answer or not answer.startswith("FAILED:"):
        return False
    if not config_path or _uses_local_llm(config_path):
        return False
    if "422" not in answer:
        return False
    lower = answer.lower()
    if any(term in lower for term in _NOT_RECOVERABLE):
        return False
    return True


def _recover_and_retry(messages, config_path, timeout=300):
    """Auto-recover from cloud 422: cooldown, restart NAT, retry once.

    Returns the new answer string, or None if recovery failed.
    """
    print("  Auto-recovering: waiting 30 s for cooldown...")
    time.sleep(30)
    print("  Restarting NAT...")
    if not start_nat(config_path):
        print("  Recovery failed: could not restart NAT.")
        return None
    print("  Retrying question...")
    return ask_with_spinner(messages, timeout=timeout)


def _validate_yaml(path):
    """Validate an agent YAML before starting NAT. Returns (ok, message)."""
    if not path:
        return False, "No config path provided."
    if not os.path.exists(path):
        return False, f"File not found: {path}"
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, f"YAML parse error: {e}"
    if not isinstance(data, dict):
        return False, "YAML did not parse to a dictionary."
    wf = data.get("workflow")
    if not wf or not isinstance(wf, dict):
        return False, "Missing or invalid 'workflow' section."
    wf_type = wf.get("_type", "")
    valid_types = {
        "tool_calling_agent", "react_agent", "reasoning_agent",
        "rewoo_agent", "router_agent",
        "sequential_executor", "parallel_executor",
    }
    if wf_type not in valid_types:
        return False, (
            f"workflow._type is '{wf_type}'. "
            f"Expected one of: {', '.join(sorted(valid_types))}"
        )
    llm_key = wf.get("llm_name")
    if not llm_key:
        return False, "workflow.llm_name is missing."
    llms = data.get("llms", {})
    if llm_key not in llms:
        return False, (
            f"workflow.llm_name '{llm_key}' not found in llms section. "
            f"Available: {', '.join(llms.keys()) or 'none'}"
        )
    return True, "Config is valid."


def parse_agent_info(config_path):
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
        wf = data.get("workflow", {})
        wf_type = wf.get("_type", "unknown")
        llm_key = wf.get("llm_name", "unknown")
        llms = data.get("llms", {})
        llm_config = llms.get(llm_key, {})
        model_name = llm_config.get("model_name", llm_key)
        funcs = data.get("functions", {})
        agents = [k for k, v in funcs.items()
                  if isinstance(v, dict) and "agent" in str(v.get("_type", ""))]
        tools = [k for k in funcs if k not in agents]
        temperature = llm_config.get("temperature")
        seed = llm_config.get("seed")
        max_tokens = llm_config.get("max_tokens")
        freq_penalty = llm_config.get("frequency_penalty")
        return {
            "type": wf_type,
            "model": model_name,
            "tools": tools,
            "sub_agents": agents,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
            "frequency_penalty": freq_penalty,
        }
    except Exception as e:
        return {"error": str(e)}


def _uses_local_llm(config_path):
    """Return True if the agent config uses a local vLLM (not a cloud NIM endpoint)."""
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
        wf = data.get("workflow", {})
        llm_key = wf.get("llm_name", "")
        llm_cfg = data.get("llms", {}).get(llm_key, {})
        if llm_cfg.get("_type") == "nim":  # ChatNVIDIA defaults to cloud
            return False
        base_url = llm_cfg.get("base_url", "")
        if base_url and "localhost" not in base_url and "127.0.0.1" not in base_url:
            return False
        return True
    except Exception:
        return True


_AGENT_DESIGNS = {
    "single": (
        "I'm the single agent: one tool_calling_agent with direct access to all 11 tools.\n"
        "  The LLM decides which tool to call based on a minimal system prompt.\n"
        "  No routing logic, no sub-agents, no orchestrator. The simplest architecture."
    ),
    "multi": (
        "I'm the multi agent: a tool_calling_agent orchestrator with 3 specialist sub-agents.\n"
        "  I classify each question, delegate to ONE specialist, then relay the answer.\n"
        "  Each sub-agent has its own tools and prompt. Extra LLM call for routing,\n"
        "  but specialists are isolated and focused."
    ),
    "ultrafast": (
        "I'm the ultrafast agent: a tool_calling_agent with embedded routing in the system prompt.\n"
        "  A TYPE A/B/C/D decision tree classifies questions before any tool call.\n"
        "  Same flat architecture as single, but prompt-driven routing eliminates\n"
        "  the orchestrator LLM call. One agent, all tools, one LLM call per step."
    ),
    "ultrafast-nogpu": (
        "I'm the ultrafast-nogpu agent: same ultrafast architecture, but inference\n"
        "  runs on NVIDIA Build instead of a local GPU. No vLLM, no model download,\n"
        "  no VRAM needed. Model: Qwen 3.5-122B-A10B (cloud). Same tools and routing prompt.\n"
        "  Trade-offs: higher latency per question, 4096 max output tokens\n"
        "  (vs 16384 local), and uses NVIDIA Build credits. Retries are automatic."
    ),
}


def _build_custom_design(config_path):
    """Generate a dynamic description for a custom/unknown agent from its YAML."""
    info = parse_agent_info(config_path)
    if "error" in info:
        return f"I'm a custom agent loaded from {config_path}."
    wf_type = info["type"]
    n_tools = len(info["tools"])
    n_subs = len(info["sub_agents"])
    if n_subs:
        sub_list = ", ".join(info["sub_agents"])
        return (
            f"I'm a custom {wf_type} with {n_subs} sub-agent(s): {sub_list}.\n"
            f"  {n_tools} tool(s) are registered across the agent hierarchy.\n"
            f"  Loaded from {config_path}."
        )
    return (
        f"I'm a custom {wf_type} with direct access to {n_tools} tool(s).\n"
        f"  Loaded from {config_path}."
    )


def _print_agent_info(agent_name, config_path):
    """Print deterministic agent identity card."""
    design = _AGENT_DESIGNS.get(agent_name)
    if design is None:
        design = _build_custom_design(config_path)
    print(f"\n  {design}\n")
    print(f"  Name:       {agent_name}")
    print(f"  Config:     {config_path}")
    if config_path and os.path.exists(config_path):
        info = parse_agent_info(config_path)
        if "error" not in info:
            print(f"  Model:      {info['model']}")
            temp = info.get('temperature')
            seed = info.get('seed')
            max_tok = info.get('max_tokens')
            freq_pen = info.get('frequency_penalty')
            params = []
            if temp is not None:
                params.append(f"temperature={temp}")
            if seed is not None:
                params.append(f"seed={seed}")
            if max_tok is not None:
                params.append(f"max_tokens={max_tok}")
            if freq_pen is not None:
                params.append(f"frequency_penalty={freq_pen}")
            if params:
                print(f"  LLM params: {', '.join(params)}")
            print(f"  Tools:      {', '.join(info['tools'])}")
            if info['sub_agents']:
                print(f"  Sub-agents: {', '.join(info['sub_agents'])}")
    if config_path:
        print(f"  Full YAML:  cat {config_path}")
    print("\n  Ask me anything, or type 'help' for commands.")


_IDENTITY_PATTERNS = re.compile(
    r"who are you|what are you|what (type|kind) of agent"
    r"|describe yourself|tell me about you"
    r"|what tools do you have|what model are you"
    r"|what is your (config|architecture|model|tools)",
    re.IGNORECASE,
)


_AGENT_BACKEND_LABELS = {
    "single": "simplest: 1 agent, all tools, local GPU",
    "multi": "3 specialist sub-agents + orchestrator, local GPU",
    "ultrafast": "fastest: prompt-driven routing, local GPU",
    "ultrafast-nogpu": "no GPU needed: ultrafast on NVIDIA Build",
}


def pick_agent(default="ultrafast"):
    default_idx = list(AGENTS.keys()).index(default) + 1 if default in AGENTS else 3
    print("\n  Available agents:")
    col = max(len(n) for n in AGENTS) + 2
    for i, (name, path) in enumerate(AGENTS.items(), 1):
        tag = " (default)" if i == default_idx else ""
        label = _AGENT_BACKEND_LABELS.get(name, path)
        print(f"    {i}. {name:<{col}}{label}{tag}")
    print(f"    {len(AGENTS)+1}. {'custom':<{col}}your own YAML config")

    while True:
        try:
            choice = input(f"\n  Pick agent (or Enter for {default}): ").strip() or str(default_idx)
        except (EOFError, KeyboardInterrupt):
            return None, None

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(AGENTS):
                name = list(AGENTS.keys())[idx - 1]
                return name, AGENTS[name]
            elif idx == len(AGENTS) + 1:
                path = input("  Path to your agent YAML (e.g., my-agent/gaia_agent.yml): ").strip()
                if not path or not os.path.exists(path):
                    print("  File not found.")
                    continue
                ok, msg = _validate_yaml(path)
                if not ok:
                    print(f"  Invalid config: {msg}")
                    continue
                return "custom", path
        elif choice in AGENTS:
            return choice, AGENTS[choice]
        elif os.path.exists(choice):
            ok, msg = _validate_yaml(choice)
            if not ok:
                print(f"  Invalid config: {msg}")
                continue
            return "custom", choice

        print("  Invalid choice.")


def start_nat(config_path):
    try:
        result = subprocess.run(
            ["bash", "gaia_tools/gaia_run.sh", "-c", config_path, "--serve"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            return True
        else:
            detail = (result.stderr or result.stdout or "").strip()
            if detail:
                print(f"  NAT failed to start: {detail[-500:]}")
            else:
                print(f"  NAT failed to start (exit code {result.returncode}).")
            return False
    except subprocess.TimeoutExpired:
        print("  NAT startup timed out after 120s.")
        return False
    except Exception as e:
        print(f"  Error starting NAT: {e}")
        return False


def _print_phoenix_hint():
    """Print how to view Phoenix based on local vs remote environment."""
    print("  View traces: http://localhost:6006 (if browser is on this machine)")
    if IS_REMOTE:
        print("  If your browser is on a different machine (e.g., your laptop):")
        print("    1. Open a NEW terminal on your laptop (not here)")
        print("    2. Run: ssh -L 6006:localhost:6006 <same-host-you-ssh-into>")
        print("    3. Then open http://localhost:6006 in your browser")


def start_phoenix():
    print("\n  Starting Phoenix tracing...")
    try:
        subprocess.run(
            ["bash", "gaia_tools/start_services.sh", "--phoenix"],
            capture_output=True, text=True, timeout=60
        )
        if check_service(PHOENIX_HEALTH):
            print("  Phoenix is running on port 6006.")
            _print_phoenix_hint()
            return True
        else:
            print("  Phoenix failed to start. Non-critical, agent still works.")
            return False
    except Exception as e:
        print(f"  Phoenix error: {e}")
        return False


COMMANDS_PRIMARY = [
    ("<any text>",       "Ask anything (multi-turn, remembers context)"),
    ("level <L>, <N>",   "Run GAIA question N from Level L"),
    ("benchmark",        "Run 20-question HF leaderboard"),
    ("switch",           "Change agent"),
    ("tracing",          "Open Phoenix traces in your browser"),
    ("status",           "Services and API key health"),
    ("help",             "All commands"),
    ("quit",             "Exit (resume with ./ask)"),
]

def _print_commands():
    """Startup: short list with an inviting intro."""
    print("  Commands:")
    for cmd, desc in COMMANDS_PRIMARY:
        print(f"    {cmd:20s}{desc}")
    print()


def _print_help():
    """Full command reference, grouped by whitespace."""
    COL = 20
    print()
    print(f"    {'level':{COL}}GAIA validation set summary (all levels)")
    print(f"    {'level <L>':{COL}}Show Level L questions (1, 2, or 3)")
    print(f"    {'level <L>, <N>':{COL}}Run question N from Level L (e.g., level 1, 3)")
    print(f"    {'benchmark [agent]':{COL}}Run 20-question scored leaderboard")
    print()
    print(f"    {'switch [agent]':{COL}}Change agent (clears conversation)")
    print(f"    {'clear':{COL}}Reset conversation memory")
    print(f"    {'info':{COL}}Agent, model, and tools")
    print(f"    {'verbose on/off':{COL}}Show/hide full model reasoning")
    print()
    print(f"    {'status':{COL}}Services and API key health")
    print(f"    {'tracing':{COL}}Open Phoenix traces in browser")
    print(f"    {'quit':{COL}}Exit (services keep running)")
    print()
    print("  Agents: single, multi, ultrafast, ultrafast-nogpu, custom")
    print()


def _vllm_status(config_path):
    """Return vLLM display string, reflecting actual health and current agent needs."""
    is_local = _uses_local_llm(config_path)
    vllm_up = check_service(VLLM_HEALTH)
    if is_local:
        return "OK" if vllm_up else "DOWN"
    return "OK (unused)" if vllm_up else "off (not needed)"


def print_status_line(agent_name, verbose=True, config_path=None):
    vllm = _vllm_status(config_path) if config_path else ("OK" if check_service(VLLM_HEALTH) else "DOWN")
    nat = "OK" if check_service(NAT_HEALTH) else "DOWN"
    phoenix_ok = check_service(PHOENIX_HEALTH)
    phoenix = "OK" if phoenix_ok else "off"
    v = "ON" if verbose else "OFF"
    print()
    print(f"  Agent: {agent_name} | vLLM: {vllm} | NAT: {nat} | "
          f"Phoenix: {phoenix} | Verbose: {v}")
    print()
    _print_commands()
    if nat == "OK":
        print("  Ready. Type a question, or 'help' for commands.")
    else:
        print("  Setup incomplete. Type 'status' to diagnose.")


def print_status_compact(agent_name, verbose=True, config_path=None):
    vllm = _vllm_status(config_path) if config_path else ("OK" if check_service(VLLM_HEALTH) else "DOWN")
    nat = "OK" if check_service(NAT_HEALTH) else "DOWN"
    phoenix = "OK" if check_service(PHOENIX_HEALTH) else "off"
    v = "ON" if verbose else "OFF"
    print(f"\n  Agent: {agent_name} | vLLM: {vllm} | NAT: {nat} | "
          f"Phoenix: {phoenix} | Verbose: {v}\n")


LEVEL_INFO = {
    "1": ("Easy",   "1-2 tool calls, direct answers"),
    "2": ("Medium", "Multi-step reasoning, may involve files"),
    "3": ("Hard",   "Long reasoning chains, complex file analysis"),
}


def _build_level_index(questions):
    """Build a mapping from (level, within-level position) to flat index.

    Returns dict: {"1": [(flat_idx, question), ...], "2": [...], ...}
    """
    by_level = {}
    for flat_idx, q in enumerate(questions):
        lvl = str(q.get("level", q.get("Level", "?")))
        by_level.setdefault(lvl, []).append((flat_idx, q))
    return by_level


def _level_counts(questions):
    """Return dict of level -> count."""
    from collections import Counter
    return Counter(str(q.get("level", q.get("Level", "?"))) for q in questions)


def _level_summary(questions):
    """Formatted level summary table for terminal output."""
    counts = _level_counts(questions)
    parts = []
    for lvl in sorted(counts):
        diff, desc = LEVEL_INFO.get(lvl, ("", ""))
        parts.append(f"    L{lvl}  {counts[lvl]:3d} questions   {diff:6s}  {desc}")
    return "\n".join(parts)


def cmd_level(questions, level_str=None):
    """Show level summary, or list questions for a specific level."""
    if not questions:
        print("  No GAIA questions loaded. Run prep_gaia_data.py first.")
        return

    by_level = _build_level_index(questions)

    if level_str:
        items = by_level.get(level_str, [])
        if not items:
            available = sorted(by_level.keys())
            print(f"  No Level {level_str} questions. "
                  f"Available levels: {', '.join(available)}")
            return
        diff, desc = LEVEL_INFO.get(level_str, ("", ""))
        header = f"  Level {level_str}"
        if diff:
            header += f" / {diff}"
        if desc:
            header += f": {desc}"
        print(f"\n{header}")
        print(f"  {len(items)} validation questions (with expected answers):\n")
        for pos, (flat_idx, q) in enumerate(items, 1):
            full = " ".join(q.get("question", q.get("input", "")).split())
            text = full[:85] + ("..." if len(full) > 85 else "")
            has_file = " [file]" if q.get("file_name") else ""
            print(f"  {pos:3d}. {text}{has_file}")
        print(f"\n  Run a question: level {level_str}, n  "
              f"(e.g., level {level_str}, 1)")
    else:
        print(f"\n  GAIA validation set: {len(questions)} questions across "
              f"{len(by_level)} difficulty levels:\n")
        print(_level_summary(questions))
        print("\n  These are validation questions with expected answers.")
        print("  For the scored leaderboard, use: benchmark")
        print("\n  Show:  level 1, level 2, level 3")
        print("  Run:   level 1, 3  (Level 1, question 3)")


def _resolve_gaia(questions, level_str, pos):
    """Resolve (level, position) to flat index. Returns (flat_idx, q) or (None, None)."""
    by_level = _build_level_index(questions)
    items = by_level.get(level_str, [])
    if not items:
        available = sorted(by_level.keys())
        print(f"  No Level {level_str} questions. "
              f"Available: {', '.join('L' + a for a in available)}")
        return None, None
    if pos < 1 or pos > len(items):
        print(f"  Level {level_str} has {len(items)} questions. "
              f"Use 1-{len(items)} (see 'level {level_str}').")
        return None, None
    flat_idx, q = items[pos - 1]
    return flat_idx, q


def cmd_gaia(questions, level_str, pos, verbose, config_path=None):
    """Run a GAIA question by level and position. Returns (prompt, answer)."""
    if not questions:
        print("  No GAIA questions loaded.")
        return None, None

    flat_idx, q = _resolve_gaia(questions, level_str, pos)
    if q is None:
        return None, None

    expected = q.get("Final answer", q.get("expected_answer", q.get("answer", "")))
    text = q.get("question", q.get("input", ""))
    file_name = q.get("file_name", "")

    actual_level = str(q.get("level", q.get("Level", "?")))
    diff = LEVEL_INFO.get(actual_level, ("", ""))[0]
    diff_tag = f" / {diff}" if diff else ""
    print(f"\n  Question L{actual_level}.{pos} (Level {actual_level}{diff_tag})")
    print(f"  {text}")
    if file_name:
        file_exists = (GAIA_FILES_DIR / file_name).exists()
        status = "available" if file_exists else "MISSING"
        print(f"  File: {file_name} ({status})")
    print()

    if not check_service(NAT_HEALTH):
        print("  NAT is not running. Use 'switch' to start an agent.")
        return None, None

    prompt = build_question_prompt(q)
    msgs = [{"role": "user", "content": prompt}]
    answer = ask_with_spinner(msgs)

    if _should_auto_recover(answer, config_path):
        print(f"\n  {answer}")
        recovered = _recover_and_retry(msgs, config_path)
        if recovered and not recovered.startswith("FAILED:"):
            answer = recovered
        else:
            print(f"  Retry also failed: {recovered or 'NAT restart failed'}")

    if answer.startswith("FAILED:"):
        print(f"\n  {answer}")
        if config_path and not _uses_local_llm(config_path):
            print("  Hint: check NGC_API_KEY with 'status', or wait 60s and retry.")
        elif not check_service(VLLM_HEALTH):
            print("  Hint: vLLM is down. Type 'switch ultrafast-nogpu' to continue without it.")
            print("  To restart vLLM: quit, run 'bash gaia_tools/start_services.sh', then './ask'")
        return prompt, answer

    if verbose:
        print(f"\n  Full response:\n  {answer}")
    else:
        display = answer[:300] + ("..." if len(answer) > 300 else "")
        print(f"\n  Answer: {display}")

    if expected:
        extracted = _extract_final_answer(answer)
        exp = str(expected)
        if _gaia_check:
            match = _gaia_check(extracted, exp)
        else:
            norm_ans = normalize_for_comparison(extracted)
            norm_exp = normalize_for_comparison(exp)
            match = norm_ans == norm_exp or (len(norm_exp) > 2 and norm_exp in norm_ans)
        tag = "MATCH" if match else "MISMATCH"
        print(f"\n  Expected:  {expected}")
        print(f"  Submitted: {extracted}")
        print(f"  Check:     [{tag}] (approximate local check, HF is authoritative)")

    return prompt, answer


BENCHMARK_LOCK = ROOT / ".benchmark.lock"


def _pid_alive(pid):
    """Cross-platform check if a PID is still running (works on Linux and macOS)."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _acquire_benchmark_lock():
    """Try to acquire the benchmark lock. Returns True if acquired, False if busy."""
    if BENCHMARK_LOCK.exists():
        try:
            pid = int(BENCHMARK_LOCK.read_text().strip())
            if _pid_alive(pid):
                print(f"  A benchmark is already running (pid {pid}).")
                print("  Wait for it to finish, or kill it: kill " + str(pid))
                return False
        except (ValueError, OSError):
            pass
        BENCHMARK_LOCK.unlink(missing_ok=True)

    try:
        fd = os.open(str(BENCHMARK_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        print("  A benchmark just started in another window. Try again shortly.")
        return False


def _release_benchmark_lock():
    """Release the benchmark lock if we own it."""
    try:
        pid = int(BENCHMARK_LOCK.read_text().strip())
        if pid == os.getpid():
            BENCHMARK_LOCK.unlink(missing_ok=True)
    except Exception:
        BENCHMARK_LOCK.unlink(missing_ok=True)


def is_benchmark_running():
    """Check if a benchmark is currently running (used by switch to protect NAT)."""
    if not BENCHMARK_LOCK.exists():
        return False
    try:
        pid = int(BENCHMARK_LOCK.read_text().strip())
        return _pid_alive(pid)
    except Exception:
        return False


def _run_benchmark(cmd_args):
    """Run gaia_run.sh with the given arguments, handle Ctrl+C."""
    if not _acquire_benchmark_lock():
        return

    try:
        subprocess.run(["bash", "gaia_tools/gaia_run.sh"] + cmd_args)
        print("\n  Benchmark complete. Check your score: bash gaia_tools/gaia_run.sh --history")
    except KeyboardInterrupt:
        print("\n  Benchmark interrupted. Back to ask> prompt.")
    finally:
        _release_benchmark_lock()


def _run_benchmark_all():
    """Run all 3 local agents sequentially, with the same lock as single-agent runs."""
    config_path = AGENTS.get("single", "single-agent/gaia_agent.yml")
    status = _vllm_status(config_path)
    if status != "OK":
        print("  vLLM is not running. 'all' requires local vLLM for all 3 agents.")
        print("  Try: benchmark ultrafast-nogpu (works without vLLM)")
        print("  To start vLLM: quit, run 'bash gaia_tools/start_services.sh', then './ask'")
        return

    if not _acquire_benchmark_lock():
        return

    print("  Running single + multi + ultrafast sequentially (20 HF questions each).")
    print("  This takes ~60 min total. Results go to each agent's runs/latest/ folder.")
    try:
        subprocess.run(["bash", "gaia_tools/gaia_run_all.sh"])
        print("\n  Benchmark complete. Check your score: bash gaia_tools/gaia_run.sh --history")
    except KeyboardInterrupt:
        print("\n  Benchmark interrupted. Back to ask> prompt.")
    finally:
        _release_benchmark_lock()


def _benchmark_custom(questions=None, path=""):
    """Run benchmark with a custom YAML config."""
    if not path:
        try:
            path = input("  Path to your agent YAML (e.g., my-agent/gaia_agent.yml): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return
    if not path or not os.path.exists(path):
        print(f"  File not found: {path}")
        return
    ok, msg = _validate_yaml(path)
    if not ok:
        print(f"  Invalid config: {msg}")
        return
    if not _uses_local_llm(path):
        print(f"  Starting benchmark with custom config: {path} (20 HF leaderboard questions)")
        print("  Note: NVIDIA Build (NAT auto-retry enabled). Uses ~500+ credits.")
    else:
        print(f"  Starting benchmark with custom config: {path} (20 HF leaderboard questions)")
    _run_benchmark(["-c", path])


def cmd_benchmark(questions=None, arg=""):
    mode_map = {"single": "--single", "multi": "--multi",
                "ultrafast": "--ultrafast"}

    choice = arg.strip() if arg else ""
    choice_lower = choice.lower()

    if choice_lower in mode_map:
        print(f"  Starting benchmark ({choice_lower}), 20 questions from HF leaderboard...")
        _run_benchmark([mode_map[choice_lower]])
        return
    if choice_lower == "ultrafast-nogpu":
        print("  Starting benchmark (ultrafast-nogpu), 20 questions from HF leaderboard...")
        print("  Note: Qwen 3.5-122B on NVIDIA Build (retries enabled).")
        print("  Uses ~500+ credits. Check balance: https://build.nvidia.com/settings/api-keys")
        _run_benchmark(["-c", AGENTS["ultrafast-nogpu"]])
        return
    if choice_lower == "all":
        _run_benchmark_all()
        return
    if choice_lower == "custom":
        _benchmark_custom(questions)
        return
    if choice and os.path.exists(choice):
        _benchmark_custom(questions, choice)
        return

    print("\n  Run the GAIA benchmark (20 questions from HF leaderboard).")
    print("  Pick an agent:")
    print("    1. single            simplest: 1 agent, all tools, local GPU")
    print("    2. multi             3 specialist sub-agents + orchestrator, local GPU")
    print("    3. ultrafast         fastest: prompt-driven routing, local GPU")
    print("    4. ultrafast-nogpu   no GPU needed: ultrafast on NVIDIA Build")
    print("    5. all local         single + multi + ultrafast, requires local GPU")
    print("    6. custom            your own YAML config")
    print("    7. cancel")

    try:
        pick = input("\n  Agent [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return

    num_map = {"1": "single", "2": "multi", "3": "ultrafast",
               "4": "ultrafast-nogpu", "5": "all", "6": "custom", "7": "cancel"}
    pick = num_map.get(pick, pick)

    if pick == "cancel":
        print("  Cancelled.")
    elif pick in mode_map:
        print(f"  Starting benchmark ({pick}), 20 questions from HF leaderboard...")
        _run_benchmark([mode_map[pick]])
    elif pick == "ultrafast-nogpu":
        print("  Starting benchmark (ultrafast-nogpu), 20 questions from HF leaderboard...")
        print("  Note: Qwen 3.5-122B on NVIDIA Build (retries enabled).")
        print("  Uses ~500+ credits. Check balance: https://build.nvidia.com/settings/api-keys")
        _run_benchmark(["-c", AGENTS["ultrafast-nogpu"]])
    elif pick == "all":
        _run_benchmark_all()
    elif pick == "custom":
        _benchmark_custom(questions)
    else:
        print(f"  Invalid choice: {pick}")


def main():
    if sys.stdin and sys.stdin.isatty():
        if "libedit" in (getattr(readline, "__doc__", "") or ""):
            readline.parse_and_bind("bind ^[[A ed-prev-history")
            readline.parse_and_bind("bind ^[[B ed-next-history")
        else:
            readline.parse_and_bind(r'"\e[A": previous-history')
            readline.parse_and_bind(r'"\e[B": next-history')
        try:
            readline.read_history_file(HISTORY_FILE)
        except (FileNotFoundError, OSError):
            try:
                HISTORY_FILE.unlink(missing_ok=True)
            except Exception:
                pass
        readline.set_history_length(500)

    _normalize_env()

    questions = load_gaia_questions()
    if not questions:
        print("\n  No GAIA questions found. Attempting to download...\n")
        try:
            result = subprocess.run(
                [sys.executable, "gaia_tools/prep_gaia_data.py"],
                timeout=120
            )
            if result.returncode == 0:
                questions = load_gaia_questions()
                if questions:
                    print(f"\n  Downloaded {len(questions)} GAIA questions.")
                else:
                    print("\n  Download ran but no questions file produced.")
            else:
                print("\n  GAIA download failed. The 'level' command won't work.")
                print("  Accept the dataset terms at: "
                      "https://huggingface.co/datasets/gaia-benchmark/GAIA")
                print("  Then retry: python3 gaia_tools/prep_gaia_data.py")
        except subprocess.TimeoutExpired:
            print("\n  GAIA download timed out after 120s.")
            print("  Run manually: python3 gaia_tools/prep_gaia_data.py")
        except Exception as e:
            print(f"\n  Could not download GAIA data: {e}")
            print("  Run manually: python3 gaia_tools/prep_gaia_data.py")
        print()

    # Check API keys: prompt for missing, live-validate present ones
    print("  Verifying API keys:")
    api_keys = _load_api_keys()
    key_checks = [
        ("TAVILY_API_KEY", "Internet search will fail.",
         "https://tavily.com/", _check_tavily),
        ("NGC_API_KEY", "NVIDIA Build models will be unavailable.",
         "https://build.nvidia.com/", _check_ngc),
        ("HF_TOKEN", "GAIA leaderboard submission may fail.",
         "https://huggingface.co/settings/tokens", _check_hf),
    ]
    is_tty = sys.stdin and sys.stdin.isatty()
    for key_name, impact, url, checker in key_checks:
        label = KEY_DISPLAY.get(key_name, key_name)
        prefix = KEY_PREFIXES.get(key_name, "")
        val = api_keys.get(key_name, "")
        if not val:
            print(f"\n    {label}: NOT SET. {impact}")
            print(f"    Get one at: {url}")
            if is_tty:
                try:
                    val = input(f"  Paste {key_name} (or Enter to skip): ").strip()
                except (EOFError, KeyboardInterrupt):
                    val = ""
                if val:
                    if prefix and not val.startswith(prefix):
                        print(f"  Warning: {key_name} usually starts with '{prefix}'. Double-check it.")
                    if _save_api_key(key_name, val):
                        api_keys[key_name] = val
                        print("  Saved to .env")
                        verify = checker(val)
                        print(f"  {label}: {verify}")
                    else:
                        print("  Could not save to .env. Add it manually.")
        else:
            print(f"    {label}: checking...", end="", flush=True)
            key_status = checker(val)
            print(f"\r    {label}: {key_status:<15}")
            if key_status not in ("OK", "TIMEOUT", "UNREACHABLE"):
                print(f"  {impact}")
                print(f"  Get a new key at: {url}")
                if is_tty:
                    try:
                        new_val = input(f"  Paste new {key_name} (or Enter to skip): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        new_val = ""
                    if new_val:
                        if prefix and not new_val.startswith(prefix):
                            print(f"  Warning: {key_name} usually starts with '{prefix}'. Double-check it.")
                        if _save_api_key(key_name, new_val):
                            api_keys[key_name] = new_val
                            print("  Saved to .env")
                            verify = checker(new_val)
                            print(f"  {label}: {verify}")
                        else:
                            print("  Could not save to .env. Add it manually.")

    # Pick default agent: prefer ultrafast (local), but auto-switch to
    # ultrafast-nogpu if vLLM is not running and the config exists.
    vllm_up = check_service(VLLM_HEALTH)
    cloud_available = Path(AGENTS.get("ultrafast-nogpu", "")).exists()

    if vllm_up or not cloud_available:
        agent_name = "ultrafast"
        config_path = AGENTS[agent_name]
        if not vllm_up:
            print("\n  vLLM not running. Type 'switch ultrafast-nogpu', or quit and run 'bash gaia_tools/start_services.sh'.")
    else:
        agent_name = "ultrafast-nogpu"
        config_path = AGENTS[agent_name]

    if _uses_local_llm(config_path):
        backend = "local vLLM"
    else:
        backend = "NVIDIA Build, no local GPU"
    print(f"\n  Starting NeMo Agent Toolkit (NAT) with {agent_name} agent (default, {backend})...")
    print(f"  Type 'switch' to change agent, or 'info' to see current config.")
    if not start_nat(config_path):
        if check_service(NAT_HEALTH):
            print("  Using previously running agent. Try 'switch' to change.")
        else:
            print("  NAT failed to start. Use 'switch' to try again.")

    verbose = True
    conversation = []
    config_ctx = build_config_context(config_path)
    print_status_line(agent_name, verbose, config_path)

    while True:
        turn = len(conversation) // 2
        prompt_str = f"ask [{turn}]> " if turn > 0 else "ask> "
        try:
            line = input(prompt_str).strip()
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print()
            break

        if not line:
            continue

        try:
            readline.write_history_file(HISTORY_FILE)
        except Exception:
            pass

        parts = line.split(None, 1)
        cmd = parts[0].lower()
        # Standalone commands: match only when the entire line is the command.
        # This prevents "clear explanation" or "help me" from triggering commands.
        line_lower = line.lower()

        if line_lower in ("quit", "exit", "q"):
            print("  Bye! Resume with ./ask (starts fresh with default agent, services keep running).")
            break

        elif line_lower == "help":
            _print_help()

        elif line_lower == "clear":
            conversation.clear()
            print("  Conversation cleared.")

        elif line_lower == "status":
            vllm_display = _vllm_status(config_path)
            nat_ok = check_service(NAT_HEALTH)
            phoenix_ok = check_service(PHOENIX_HEALTH)
            print(f"  vLLM:    {vllm_display}")
            print(f"  NAT:     {'OK' if nat_ok else 'DOWN'}")
            print(f"  Phoenix: {'OK' if phoenix_ok else 'off'}")
            if phoenix_ok:
                _print_phoenix_hint()
            keys = _load_api_keys()
            needs_fix = []
            status_checks = [
                ("TAVILY_API_KEY", _check_tavily,
                 "internet search will fail", "https://tavily.com/"),
                ("NGC_API_KEY", _check_ngc,
                 "NVIDIA Build models will be unavailable", "https://build.nvidia.com/"),
                ("HF_TOKEN", _check_hf,
                 "leaderboard submission may fail",
                 "https://huggingface.co/settings/tokens"),
            ]
            for key_name, checker, impact, url in status_checks:
                label = KEY_DISPLAY.get(key_name, key_name)
                val = keys.get(key_name, "")
                if val:
                    print(f"  {label}: checking...", end="", flush=True)
                    key_status = checker(val)
                    print(f"\r  {label}: {key_status:<15}")
                    if key_status not in ("OK", "TIMEOUT", "UNREACHABLE"):
                        needs_fix.append((key_name, checker, key_status,
                                          impact, url))
                else:
                    print(f"  {label}: NOT SET ({impact})")
                    needs_fix.append((key_name, checker, "NOT SET",
                                      impact, url))
            any_replaced = False
            if needs_fix and is_tty:
                for key_name, fix_checker, reason, impact, url in needs_fix:
                    label = KEY_DISPLAY.get(key_name, key_name)
                    prefix = KEY_PREFIXES.get(key_name, "")
                    print(f"\n  {label}: {reason}")
                    print(f"  Get one at: {url}")
                    try:
                        val = input(f"  Paste {key_name} (or Enter to skip): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        val = ""
                    if val:
                        if prefix and not val.startswith(prefix):
                            print(f"  Warning: {key_name} usually starts with "
                                  f"'{prefix}'. Double-check it.")
                        if _save_api_key(key_name, val):
                            print("  Saved to .env")
                            verify = fix_checker(val)
                            print(f"  {label}: {verify}")
                            any_replaced = True
                        else:
                            print("  Could not save. Add it to .env manually.")
            elif needs_fix:
                print("\n  To fix: set keys in .env and restart with ./ask")
            if any_replaced:
                print("\n  Keys updated in .env. Type 'switch' to restart "
                      "the agent with the new keys.")

        elif line_lower == "info":
            _print_agent_info(agent_name, config_path)

        elif cmd == "level":
            raw_args = line[len("level"):].replace(",", " ").split()
            if not raw_args:
                cmd_level(questions)
            elif len(raw_args) == 1 and raw_args[0] in ("1", "2", "3"):
                cmd_level(questions, level_str=raw_args[0])
            elif len(raw_args) == 2 and raw_args[0] in ("1", "2", "3") and raw_args[1].isdigit():
                lvl, pos = raw_args[0], int(raw_args[1])
                try:
                    prompt, answer = cmd_gaia(questions, lvl, pos, verbose, config_path)
                    if prompt and answer and not answer.startswith("FAILED:"):
                        conversation.clear()
                        conversation.append({"role": "user", "content": prompt})
                        conversation.append({"role": "assistant", "content": answer})
                except KeyboardInterrupt:
                    print("\n  Interrupted.")
            else:
                print("  Usage:")
                print("    level         Show level summary")
                print("    level 1       Show Level 1 questions")
                print("    level 1, 3    Run question 3 from Level 1")

        elif line_lower in ("tracing", "phoenix"):
            try:
                start_phoenix()
            except KeyboardInterrupt:
                print("\n  Phoenix start cancelled.")

        elif cmd == "switch":
            if is_benchmark_running():
                print("  Cannot switch agents while a benchmark is running.")
                print("  Wait for it to finish, or cancel it first.")
                continue
            raw_arg = parts[1].strip() if len(parts) > 1 else ""
            arg_lower = raw_arg.lower()
            if arg_lower in AGENTS:
                new_name, new_path = arg_lower, AGENTS[arg_lower]
            elif raw_arg and os.path.exists(raw_arg):
                new_name, new_path = "custom", raw_arg
            else:
                new_name, new_path = pick_agent()
            if new_name and new_path:
                if new_name == "custom":
                    ok, msg = _validate_yaml(new_path)
                    if not ok:
                        print(f"  Invalid config: {msg}")
                        continue
                if _uses_local_llm(new_path) and not check_service(VLLM_HEALTH):
                    print(f"\n  WARNING: vLLM is not running. {new_name} agent needs local vLLM.")
                    print("  Type 'switch ultrafast-nogpu' to continue without it (no GPU needed).")
                    print("  To start vLLM: quit, run 'bash gaia_tools/start_services.sh', then './ask'\n")
                try:
                    if start_nat(new_path):
                        agent_name = new_name
                        config_path = new_path
                        conversation.clear()
                        config_ctx = build_config_context(config_path)
                        print_status_compact(agent_name, verbose, config_path)
                        if not _uses_local_llm(config_path):
                            print("  Note: NVIDIA Build (no local GPU). Each question uses 1-10 API calls.")
                            print("  Simple questions: ~30-60s. Multi-step GAIA questions: ~2-3 min.")
                            print("  Rate limit: ~40 RPM. If a question fails, wait 60s and retry.")
                            print("  Credits: https://build.nvidia.com/settings/api-keys\n")
                    else:
                        print("  Switch failed. Previous agent may still be running.")
                except KeyboardInterrupt:
                    print("\n  Switch cancelled.")

        elif cmd == "verbose":
            if len(parts) > 1:
                flag = parts[1].lower()
                verbose = flag in ("on", "true", "1")
            else:
                verbose = not verbose
            print(f"  Verbose: {'ON' if verbose else 'OFF'}")

        elif cmd == "benchmark":
            cmd_benchmark(questions, parts[1].strip() if len(parts) > 1 else "")

        else:
            if _IDENTITY_PATTERNS.search(line):
                _print_agent_info(agent_name, config_path)
                continue

            if not check_service(NAT_HEALTH):
                print("  NAT is not running. Use 'switch' to start an agent.")
                continue

            conversation.append({"role": "user", "content": line})
            if len(conversation) > MAX_TURNS * 2:
                conversation = conversation[-(MAX_TURNS * 2):]

            messages = []
            if config_ctx:
                messages.append(config_ctx)
            messages.extend(conversation)

            try:
                answer = ask_with_spinner(messages)

                if _should_auto_recover(answer, config_path):
                    print(f"\n  {answer}")
                    recovered = _recover_and_retry(messages, config_path)
                    if recovered and not recovered.startswith("FAILED:"):
                        answer = recovered
                    else:
                        print(f"  Retry also failed: {recovered or 'NAT restart failed'}")

                if answer.startswith("FAILED:"):
                    conversation.pop()
                    print(f"\n  {answer}")
                    if "rate limit" in answer.lower() or "429" in answer or "422" in answer:
                        print("  Hint: check NGC_API_KEY with 'status', or wait 60s and retry.")
                    elif _uses_local_llm(config_path) and not check_service(VLLM_HEALTH):
                        print("  Hint: vLLM is down. Type 'switch ultrafast-nogpu' to continue without it.")
                        print("  To restart vLLM: quit, run 'bash gaia_tools/start_services.sh', then './ask'")
                    elif not _uses_local_llm(config_path):
                        print("  Hint: wait 60s and retry, or check NGC_API_KEY with 'status'.")
                    else:
                        print("  Hint: type 'status' for services and API key status.")
                    print()
                else:
                    conversation.append({"role": "assistant", "content": answer})
                    if verbose:
                        print(f"\n{answer}\n")
                    else:
                        display = answer[:500] + ("..." if len(answer) > 500 else "")
                        print(f"\n{display}\n")
            except KeyboardInterrupt:
                if conversation and conversation[-1]["role"] == "user":
                    conversation.pop()
                print("\n  Interrupted.")

    try:
        readline.write_history_file(HISTORY_FILE)
    except Exception:
        pass


if __name__ == "__main__":
    main()
