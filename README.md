# NAT Agent Lab

Hands-on lab for building, exploring, and benchmarking AI agents with [NVIDIA NeMo Agent Toolkit (NAT)](https://docs.nvidia.com/nemo/agent-toolkit/) on the [GAIA benchmark](https://arxiv.org/abs/2311.12983).

- **Four agent architectures** with different engineering tradeoffs (same tools, same benchmark)
- **Interactive chat** (`./ask`) with multi-turn memory, live tool tracing, and agent switching
- **GAIA validation set** (Levels 1-3) with expected answers for practice and iteration
- **GAIA benchmark** with automated [HuggingFace leaderboard](https://huggingface.co/spaces/agents-course/Students_Leaderboard) submission (20 scored questions)
- **OpenTelemetry tracing** via Phoenix to visualize every tool call and LLM step
- Works with a **local LLM via vLLM** (8x H100) or **NVIDIA Build cloud** (no GPU needed)

## Get Running

If setup is already done, just start chatting:

```bash
./ask
```

If you haven't set up yet, see [Setup](#setup) below, then come back here.

## What to Try

Once you see the `ask [1]>` prompt, try these in order:

**1. Ask a question**

```
ask [1]> What is the tallest building in San Francisco?
```

The agent searches the web, reasons over the result, and answers. Type a follow-up: the agent remembers context. The prompt shows your turn count (`ask [2]>`, `ask [3]>`, etc.).

**2. Run a GAIA validation question**

```
ask [2]> level 1, 1
```

This runs the 1st question from GAIA Level 1. The agent works through it with tool calls, then the expected answer is shown so you can compare. After it finishes, you can ask "why did you use that tool?" or "explain your reasoning" since the Q&A is seeded into memory.

Type `level 1` to see all Level 1 questions. Type `level` to see a summary of all levels.

**3. Switch agents and compare**

```
ask [3]> switch single
```

This loads the single agent and clears conversation memory (so comparisons are fair). Now re-run the same question:

```
ask [1]> level 1, 1
```

Compare: did it use the same tools? Did it get the same answer? How many LLM calls did it make? Try `switch multi` and `switch ultrafast` to see how each architecture handles the same question differently.

**4. Look at the traces**

```
ask [2]> tracing
```

This opens Phoenix in your browser. You can see every LLM call, tool invocation, and agent step visualized as a trace. Compare traces across agents to understand the architectural differences.

**5. Run the benchmark**

```
ask [1]> benchmark
```

This runs 20 scored questions from the HF leaderboard and submits your answers automatically. Your score appears on the [public leaderboard](https://huggingface.co/spaces/agents-course/Students_Leaderboard) within seconds.

**6. Build your own agent**

```bash
mkdir my-agent
cp single-agent/gaia_agent.yml my-agent/gaia_agent.yml
# Edit the YAML: change the system prompt, tools, agent type, or model
./ask
# Then: switch my-agent/gaia_agent.yml
```

Run `benchmark custom` to score your custom agent on the leaderboard.

## Agent Architectures

All four agents share the same tools: `internet_search`, `wiki_search`, `read_file`, `fetch_url`, `python_executor`, `describe_image`, `describe_image_alt`, `transcribe_audio`, `get_youtube_transcript`, `solve_chess`, `current_datetime`.

| Agent | Config | Architecture | LLM | Key difference |
|-------|--------|-------------|-----|----------------|
| **Single** | `single-agent/gaia_agent.yml` | `tool_calling_agent`, flat: LLM has direct access to all tools | MiniMax M2.5 456B MoE (local vLLM) | Simplest. One LLM call per tool step. No routing overhead. |
| **Multi** | `multi-agent/gaia_agent_multi.yml` | `tool_calling_agent` orchestrator dispatches to 3 specialist sub-agents (web, file, multimedia), each with isolated tool subsets | MiniMax M2.5 456B MoE (local vLLM) | Extra LLM call for routing, but specialists have focused prompts and tools. |
| **Ultrafast** | `ultrafast-agent/gaia_agent_ultrafast.yml` | `tool_calling_agent`, flat: embedded TYPE A/B/C/D routing decision tree in the system prompt classifies questions before any tool call | MiniMax M2.5 456B MoE (local vLLM) | Same flat architecture as Single, but prompt-driven routing eliminates the orchestrator LLM call. |
| **Ultrafast-nogpu** | `ultrafast-nogpu-agent/gaia_agent_ultrafast_nogpu.yml` | Same as Ultrafast, but LLM inference runs on NVIDIA Build instead of local vLLM | Qwen 3.5-122B-A10B MoE (NVIDIA Build) | No GPU, no model download, no VRAM. Higher latency, 4096 max output tokens, uses API credits. |

## Conversation Memory

`./ask` is **multi-turn**: the agent remembers your conversation and can answer follow-ups. The prompt shows the turn count (e.g., `ask [3]>`). Memory is kept for up to 20 turns, then the oldest turns are trimmed automatically.

Three things clear memory:

- **`clear`**: Manually reset conversation without restarting.
- **`switch`**: Changing agents always clears memory (the new agent starts fresh).
- **`level <L>, <N>`**: GAIA questions are sent **standalone** with no prior context, so the agent cannot be confused by earlier turns. After the answer, the Q&A pair is seeded into memory for follow-ups.

If a question fails (timeout, API error), the failed message is removed from memory so it does not pollute future turns.

## Commands

Type `help` in `./ask` for the full list. Key commands:

| Command | What it does |
|---------|-------------|
| `level <L>` | Show Level L questions (1, 2, or 3) |
| `level <L>, <N>` | Run question N from Level L (e.g., `level 1, 3`) |
| `benchmark [agent]` | Run 20-question scored leaderboard |
| `switch [agent]` | Pick from built-in agents or pass your own YAML (`switch my-agent/config.yml`). Clears memory. |
| `clear` | Reset conversation memory |
| `info` | Current agent, model, and tools |
| `status` | Services and API key health |
| `tracing` | Open Phoenix traces in browser |
| `verbose on/off` | Show/hide full model reasoning |

Agents: single, multi, ultrafast, ultrafast-nogpu, custom.

## Setup

### Path A: GPU instance (all 4 agents)

**What you need:**
- Linux with **8 GPUs, ~640 GB VRAM total** (e.g., 8x H100 80GB, 8x A100 80GB). Tested on GCP `a3-highgpu-8g` and Brev `8xH100`.
- **300 GB free disk space** for [MiniMax M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) model weights (~220 GB). MiniMax M2.5 is a 456B MoE model served via vLLM with `--tensor-parallel-size 8`.
- **NVIDIA drivers** installed (`nvidia-smi` should show all 8 GPUs)
- Python 3.10+ and `tmux` (pre-installed on standard GPU cloud images)

**Steps:**

```bash
git clone <repo-url> nat-agent-lab
cd nat-agent-lab
bash setup.sh                        # ~20 min (mostly model download)
bash gaia_tools/start_services.sh    # ~5-10 min (vLLM loads model into GPU memory)
./ask
```

You'll see `Agent: ultrafast | vLLM: OK | NAT: OK | Phoenix: OK | Verbose: ON`. All 4 agents are available via `switch`.

### Path B: No GPU (cloud agent only)

**What you need:**
- Linux or macOS (tested on Ubuntu 22.04, macOS 14+). No GPU, no VRAM, no model download.
- An **NGC_API_KEY** from [build.nvidia.com](https://build.nvidia.com/). This key authenticates LLM calls to NVIDIA Build ([Qwen 3.5-122B-A10B](https://build.nvidia.com/qwen/qwen3-5-122b-a10b)), rate-limited to ~40 RPM with automatic retries.
- **Credits:** 1,000 free on signup (personal email), up to 5,000 with a business or institutional email (Profile > Request More). A 20-question benchmark uses ~500+ credits. Check balance at [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys).

**Steps:**

```bash
git clone <repo-url> nat-agent-lab
cd nat-agent-lab
bash setup.sh --cloud                # skips vLLM and model download
./ask
```

You'll see `Agent: ultrafast-nogpu | vLLM: off (not needed) | NAT: OK`. Only the ultrafast-nogpu agent is available (the other 3 need local vLLM).

### API Keys

Three keys are needed for both paths (free tiers are sufficient). `setup.sh` prompts for them interactively and saves to `.env`.

| Key | Sign up | Used for |
|-----|---------|----------|
| **Tavily** | [tavily.com](https://tavily.com/) | Internet search tool |
| **NVIDIA Build** | [build.nvidia.com](https://build.nvidia.com/) | Vision/audio tools (both paths); main LLM (Path B) |
| **HuggingFace** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | GAIA dataset download and leaderboard submission |

**Coming back later?** Just run `./ask`. vLLM and Phoenix run in background tmux sessions that survive SSH disconnects. Each session starts fresh with the default agent (ultrafast if vLLM is running, ultrafast-nogpu otherwise).

## Something Not Working?

Type `status` in `./ask` for a full diagnostic (services, API keys, vLLM health). For common issues like vLLM crashes, 422 errors, disk space, or port forwarding, see [Troubleshooting](docs/GUIDE.md#troubleshooting).

## File Structure

```
.
├── setup.sh                          # One-time environment setup (--cloud for no-GPU path)
├── ask                               # Launch script (activates env, starts chat)
├── README.md                         # This file
├── docs/GUIDE.md                     # Reference guide (benchmarks, models, tracing, troubleshooting)
├── LICENSE                           # Apache 2.0 (matches upstream NAT)
├── .env.example                      # API key template
├── gaia_questions.json               # GAIA validation questions (downloaded by setup.sh)
├── gaia_files/                       # Attached files for GAIA questions
├── single-agent/
│   └── gaia_agent.yml                # Single tool_calling_agent config
├── multi-agent/
│   └── gaia_agent_multi.yml          # Multi-agent orchestrator config
├── ultrafast-agent/
│   └── gaia_agent_ultrafast.yml      # Ultrafast single-agent with routing prompt
├── ultrafast-nogpu-agent/
│   └── gaia_agent_ultrafast_nogpu.yml # ultrafast-nogpu (NVIDIA Build, no local GPU)
└── gaia_tools/
    ├── ask.py                        # Interactive chat engine (called by ./ask)
    ├── gaia_run.sh                   # One-command benchmark runner (any agent)
    ├── gaia_run_all.sh               # Run all 3 local agents sequentially
    ├── start_services.sh             # Start vLLM + Phoenix in tmux
    ├── gaia_submit.py                # Benchmark engine with answer normalization
    ├── prep_gaia_data.py             # Fetch GAIA data from HuggingFace (called by setup.sh)
    ├── pyproject.toml                # Package config for custom tools (read_file, fetch_url, etc.)
    └── src/gaia_tools/
        ├── __init__.py
        └── register.py               # Custom tool registration for NAT
```

For benchmarks, models, tracing, services, troubleshooting, and more, see **[docs/GUIDE.md](docs/GUIDE.md)**.
