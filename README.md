# NAT Agent Lab

LLM agents can search the web, execute code, read files, and reason across multiple steps, but getting them to do this reliably is an open engineering problem. This lab teaches **eval-driven agent development**: build an agent, benchmark it on real questions, inspect traces to find where it fails, improve the config, and measure again.

You will work with three components:

- **[NAT](https://github.com/NVIDIA/NeMo-Agent-Toolkit)** (NeMo Agent Toolkit): NVIDIA's open-source library for connecting and optimizing teams of AI agents. You define an agent entirely in YAML (agent type, system prompt, tools, model) and NAT handles orchestration, tool execution, and LLM calls.
- **[GAIA](https://arxiv.org/abs/2311.12983)**: a benchmark of real-world questions that require multi-step reasoning and tool use. Answers are exact-match scored, so agent output formatting matters. The repo includes the GAIA validation set (Levels 1-3) with expected answers for testing and iteration.
- **[Phoenix](https://docs.arize.com/phoenix)**: an OpenTelemetry-based tracing UI. Every LLM call, tool invocation, and routing decision is captured as a span tree you can inspect after each run.

**By the end of this lab you will know how to:**

- **Diagnose agent behavior** through traces: see every tool call, LLM prompt, and routing decision in Phoenix
- **Compare four agent topologies** on the same benchmark and understand why different routing strategies produce different scores (see [Agent Architectures](#agent-architectures))
- **Engineer better agents** by tuning system prompts, tool selection, and agent type in a single YAML config
- **Measure what matters**: submit to a [public leaderboard](https://huggingface.co/spaces/agents-course/Students_Leaderboard), iterate on your config, and track improvement

The repo includes four agents that score 85-90% on the leaderboard. They are starting points, not finished products. Study their configs, read their traces, find where they fail, and build something better.

## Quick Start

```bash
git clone <repo-url> nat-agent-lab
cd nat-agent-lab
bash setup.sh          # ~20 min; prompts for API keys, downloads model
./ask                  # start chatting
```

Complete this **before class**. Setup takes 20-30 minutes (model download, dependencies, API keys) and cannot be done during a 40-minute session. See [Setup](#setup) for Path A (GPU) vs. Path B (cloud-only) details.

You should see a status line like `Agent: ultrafast | vLLM: OK | NAT: OK | Phoenix: OK`. If any service shows a problem, type `status` for diagnostics. Ask a test question ("What is 2+2?") to confirm the agent responds. You are ready for class.

## What to Try

Once you see the `ask>` prompt, try these in order:

**1. Ask a question**

```
ask> What is the tallest building in San Francisco?
```

The agent searches the web, reasons over the result, and answers. Type a follow-up: the agent remembers context. The prompt shows your turn count (`ask [1]>` after the first exchange, `ask [2]>` after the second, etc.).

**2. Run a GAIA validation question**

```
ask [1]> level 1, 1
```

This runs the 1st question from GAIA Level 1. The agent works through it with tool calls, then the expected answer is shown so you can compare. Notice the timing: `./ask` prints how long the agent took. After it finishes, you can ask "why did you use that tool?" or "explain your reasoning" since the Q&A is seeded into memory.

Type `level 1` to see all Level 1 questions. Type `level` to see a summary of all levels.

**3. Open Phoenix (traces)**

Open Phoenix **before** running agent comparisons so traces are captured from the start.

```
ask [1]> tracing
```

This opens Phoenix in your browser. If you are on a remote machine (Brev, GCP), forward port 6006 first from a **separate terminal on your laptop**:

```bash
ssh -L 6006:localhost:6006 <your-ssh-host>
```

Then open [http://localhost:6006](http://localhost:6006). Keep Phoenix open in a browser tab. Every agent run from this point forward will appear as a trace. If Phoenix is down, agents still work normally (you just won't see the traces).

**4. Run all 4 agents on the same question and compare**

Pick a GAIA validation question (not a freeform one) so you have an expected answer to judge correctness, not just fluency. `level 1, 1` is a good choice: it requires multiple tool calls and computation, which is where architecture differences show up. Run it with each agent, note the answer, the time, and which tools were called. Switch clears memory so each comparison is fair.

You start on the default agent (ultrafast if vLLM is running). The sequence below runs all 4 agents. *(Path B users: you only have ultrafast-nogpu. Skip to step 5.)*

```
ask [1]> level 1, 1
ask [1]> switch single
ask> level 1, 1
ask [1]> switch multi
ask> level 1, 1
ask [1]> switch ultrafast-nogpu
ask> level 1, 1
```

After each run, switch to your Phoenix tab and refresh. You will see a new trace for each agent. Click into a trace to see the full agent loop: system prompt, tool calls dispatched, intermediate results, and the final answer. Compare latencies per span to find where time is spent (LLM inference vs. tool execution vs. network).

What to look for in each comparison:

| Agent | What to compare in the traces |
|-------|-------------------------------|
| **Ultrafast** *(default)* | Your baseline. Look at how the system prompt classifies the question before any tool call. |
| **Single** | No routing step. Does it call tools that Ultrafast's routing would have skipped? More or fewer LLM round-trips? |
| **Multi** | Extra LLM call for orchestrator routing. Is the routing accurate? Is the specialist's focused context worth the added latency? |
| **Ultrafast-nogpu** | Same prompt as Ultrafast, but cloud LLM. Compare network latency vs. local GPU, and answer quality across different model weights. *(Optional, requires NGC_API_KEY.)* |

**Bonus (if time permits):** run the same 4-agent comparison on a simple factual question like "What is the capital of France?" (zero tools needed, pure LLM knowledge). All 4 agents should answer instantly and correctly. If Multi is noticeably slower, that is the cost of orchestrator overhead on a question that didn't need routing.

The lesson: the best agent architecture depends on the task distribution. Flat agents have lower overhead on simple questions; routing (whether via an orchestrator LLM call or prompt-driven classification) pays off on complex, multi-tool questions. Traces make this tradeoff measurable.

**5. Run the benchmark and establish a baseline score**

The benchmark runs whichever agent is currently loaded. Switch to the one you want to score first:

```
ask [1]> switch ultrafast
ask> benchmark
```

This runs 20 scored questions (~15 min), submits your answers, and your team name appears on the [public leaderboard](https://huggingface.co/spaces/agents-course/Students_Leaderboard) within seconds. It will ask for your org and team name (e.g., `UCB` / `AgentSmiths`), and your entry shows up as `NAT-UCB-AgentSmiths-UltrafastAgent`. While it runs, go back to Phoenix and dig into the traces from step 4. This is your baseline score to beat in step 6.

**6. Build your own agent and compete**

The four included agents are baselines. Open their YAML configs, read the system prompts, and look at the traces to see where they fail. Then improve on them.

In a terminal (not inside `./ask`):

```bash
mkdir my-agent
cp ultrafast-agent/gaia_agent_ultrafast.yml my-agent/config.yml
# Edit config.yml: refine the system prompt, add/remove tools, change the agent type
```

Then load it inside `./ask`:

```
ask> switch my-agent/config.yml
```

**What to optimize** (one variable at a time, measure before and after with traces):

- **Reduce unnecessary tool calls.** Look at traces: is the agent calling `internet_search` when the answer is in the question? Add explicit instructions to the system prompt about when to use each tool and when not to. Every skipped tool call saves one LLM round-trip.
- **Remove tools the agent never uses.** Fewer tools in the YAML means a shorter tool schema in the system prompt, which means fewer input tokens per LLM call and faster inference. Check traces to see which tools are actually invoked.
- **Add prompt-driven routing.** The ultrafast agent classifies questions into TYPE A/B/C/D categories in the system prompt before making any tool call. This avoids wasted tool calls on the wrong path. Read its config to see how, and adapt the categories for your use case.
- **Change the agent type.** `tool_calling_agent` uses the LLM's native function-calling format. `react_agent` uses ReAct-style Thought/Action/Observation text prompting. These require different system prompts, so don't just swap the `_type` field. See working examples of both in the [NAT GitHub examples](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/main/examples). Copy a working `react_agent` config, adapt the tools and prompt, then compare traces: which produces fewer hallucinated tool calls? Which is faster?
- **Tune `temperature` and `seed`.** Lower temperature (0.0-0.1) makes tool selection more deterministic. Setting `seed` improves run-to-run reproducibility (local vLLM only). Measure variance by running the same `level` question 3 times.
- **Tighten the answer format.** GAIA exact-match scoring is strict. If traces show the agent is correct but formatting is wrong (extra units, commas, explanation text around the answer), add explicit formatting rules to the system prompt.
- **Reduce agent latency.** Total time = (number of LLM calls x per-call latency) + tool execution time. Traces show both. Target: reduce the number of LLM calls (routing, fewer retries) and reduce input tokens per call (shorter prompt, fewer tools).

**A note on swapping models:** each LLM has its own `max_tokens`, `temperature` range, chat template, and tool-calling format. If you change the `model` field in your YAML, you also need to adjust these parameters to match the new model's card. The built-in configs are tuned for MiniMax M2.5 (local) and Qwen 3.5-122B-A10B (cloud). See the [NAT examples](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/main/examples) for working configs with other models (Llama, Nemotron, etc.).

For deeper reading on system prompt design: [Anthropic's prompt engineering guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) covers techniques like role definition, few-shot examples, chain-of-thought, and output formatting that apply directly to agent system prompts.

**The iteration loop:** edit your YAML, run a `level` question to test, open traces to diagnose, repeat. When you're confident, run `benchmark custom` to submit your score and see it on the leaderboard.

**Compete on three dimensions:**

1. **Accuracy** (leaderboard score). The built-in agents score 85-90%. Can your custom agent beat them? Every point matters: GAIA uses exact-match scoring, so formatting and precision count.
2. **Speed** (total benchmark time). After each run, check `gaia_summary.json` for time per question and total elapsed time. A faster agent that maintains accuracy is a better agent. Compare your custom agent's timing against the built-ins.
3. **Design** (trace quality). Open your traces and a built-in agent's traces side by side. Fewer tool calls, cleaner routing, shorter prompts: these are engineering improvements you can show and explain. Be ready to present what you changed and why it worked (or didn't).

## Agent Architectures

All four agents share the same tools: `internet_search`, `wiki_search`, `read_file`, `fetch_url`, `python_executor`, `describe_image`, `describe_image_alt`, `transcribe_audio`, `get_youtube_transcript`, `solve_chess`, `current_datetime`.

| Agent | Config | Architecture | LLM | Key difference |
|-------|--------|-------------|-----|----------------|
| **Single** | `single-agent/gaia_agent.yml` | `tool_calling_agent`, flat: LLM has direct access to all tools | MiniMax M2.5 456B MoE (local vLLM) | Simplest. One LLM call per tool step. No routing overhead. |
| **Multi** | `multi-agent/gaia_agent_multi.yml` | `tool_calling_agent` orchestrator dispatches to 3 specialist sub-agents (web, file, multimedia), each with isolated tool subsets | MiniMax M2.5 456B MoE (local vLLM) | Extra LLM call for routing, but specialists have focused prompts and tools. |
| **Ultrafast** | `ultrafast-agent/gaia_agent_ultrafast.yml` | `tool_calling_agent`, flat: embedded TYPE A/B/C/D routing decision tree in the system prompt classifies questions before any tool call | MiniMax M2.5 456B MoE (local vLLM) | Same flat architecture as Single, but prompt-driven routing eliminates the orchestrator LLM call. |
| **Ultrafast-nogpu** | `ultrafast-nogpu-agent/gaia_agent_ultrafast_nogpu.yml` | Same as Ultrafast, but LLM inference runs on NVIDIA Build instead of local vLLM | Qwen 3.5-122B-A10B MoE (NVIDIA Build) | No GPU, no model download, no VRAM. Higher latency, 4096 max output tokens, uses API credits. |

Each agent is defined entirely by its YAML config: agent type, system prompt, tools, and model parameters. NAT supports additional architectures beyond `tool_calling_agent` (e.g., `react_agent`, `router_agent`, `sequential_executor`). See step 6 for how to experiment with them.

**Ultrafast-nogpu notes:** max output tokens are 4096 (vs 16384 local), so cloud responses may truncate on complex multi-step questions. `temperature: 0.0` is set, but `seed` is not supported by NVIDIA Build, and tool results (web, Wikipedia) change over time, so answers can vary between runs.

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
- **3 API keys** (free tiers are sufficient). Sign up before running setup: [Tavily](https://tavily.com/), [NVIDIA Build](https://build.nvidia.com/), [HuggingFace](https://huggingface.co/settings/tokens). See [API Keys](#api-keys) for details.

**Steps:**

```bash
git clone <repo-url> nat-agent-lab
cd nat-agent-lab
bash setup.sh                        # ~20 min; prompts for API keys, downloads model
bash gaia_tools/start_services.sh    # ~5-10 min (vLLM loads model into GPU memory)
./ask                                # verify status line shows all OK
```

You should see `Agent: ultrafast | vLLM: OK | NAT: OK | Phoenix: OK | Verbose: ON`. All 4 agents are available via `switch`, including ultrafast-nogpu (cloud LLM, no local GPU needed).

### Path B: No GPU (cloud agent only)

**What you need:**
- Linux or macOS (tested on Ubuntu 22.04, macOS 14+). No GPU, no VRAM, no model download.
- **3 API keys** (free tiers are sufficient). Sign up before running setup: [Tavily](https://tavily.com/), [NVIDIA Build](https://build.nvidia.com/), [HuggingFace](https://huggingface.co/settings/tokens). See [API Keys](#api-keys) for details.
- The **NVIDIA Build** key (NGC_API_KEY) also serves as the LLM endpoint for this path, calling [Qwen 3.5-122B-A10B](https://build.nvidia.com/qwen/qwen3-5-122b-a10b) on NVIDIA Build (rate-limited to ~40 RPM, automatic retries).
- **Credits:** 1,000 free on signup (personal email), up to 5,000 with a business or institutional email (Profile > Request More). A 20-question benchmark uses ~500+ credits. Check balance at [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys).

**Steps:**

```bash
git clone <repo-url> nat-agent-lab
cd nat-agent-lab
bash setup.sh --cloud                # ~5 min; prompts for API keys, skips model download
./ask                                # verify status line shows all OK
```

You should see `Agent: ultrafast-nogpu | vLLM: off (not needed) | NAT: OK | Phoenix: off | Verbose: ON`. Only the ultrafast-nogpu agent is available (the other 3 need local vLLM).

### API Keys

Three keys are needed for both paths (free tiers are sufficient). `setup.sh` prompts for them interactively and saves to `.env`.

| Key | Sign up | Used for |
|-----|---------|----------|
| **Tavily** | [tavily.com](https://tavily.com/) | Internet search tool |
| **NVIDIA Build** | [build.nvidia.com](https://build.nvidia.com/) | Vision/audio tools (both paths); main LLM (Path B) |
| **HuggingFace** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | GAIA dataset download and leaderboard submission |

**Coming back later?** Just run `./ask`. vLLM and Phoenix run in background tmux sessions that survive SSH disconnects. Each session starts fresh with the default agent (ultrafast if vLLM is running, ultrafast-nogpu otherwise).

## Benchmark Results

Each benchmark run saves results locally and submits to the leaderboard. Local files are saved to `<agent>/runs/runN_<config>/` (auto-numbered):

| File | What it contains |
|------|-----------------|
| `gaia_summary.json` | Score, correct count, time per question |
| `gaia_results.json` | Every question with your agent's answers |
| `benchmark.log` | Full terminal output |
| `nat.log` | NAT server log (debug tool call failures) |
| `config.yml` | Snapshot of the YAML config used |

```bash
cat ultrafast-agent/runs/latest/gaia_summary.json   # latest score
bash gaia_tools/gaia_run.sh --history                # all runs
```

You can also run benchmarks from the command line instead of `./ask`:

```bash
bash gaia_tools/gaia_run.sh --single          # single agent
bash gaia_tools/gaia_run.sh --ultrafast       # ultrafast agent
bash gaia_tools/gaia_run.sh -c my-agent/config.yml  # custom agent
```

## Troubleshooting

**vLLM won't start or crashes**
- Check GPU memory: `nvidia-smi`
- Kill stale processes: `bash gaia_tools/start_services.sh --stop`
- Verify model downloaded: `ls .cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.5/`

**NAT fails to start**
- Check port 8000: `curl localhost:8000/health`
- View NAT log: look in the run directory or `/tmp/nat_serve/`

**Phoenix not accessible from laptop**
- Verify running: `curl localhost:6006`
- Check port forwarding: `ssh -L 6006:localhost:6006 <your-ssh-host>`
- Phoenix is optional; agents work without it

**Disk space issues during setup**
- Clone to a partition with 300+ GB free (e.g., `/ephemeral/`, `/data/`)
- Or run `bash setup.sh --cloud` to skip the model download
- Check space: `df -h .`

**vLLM is DOWN after logging back in**
- vLLM runs in tmux. If the machine rebooted: `bash gaia_tools/start_services.sh`
- Check if still loading: `tmux attach -t vllm` (Ctrl+B then D to detach)

**422 errors (ultrafast-nogpu)**
- `./ask` auto-recovers: waits 30s, restarts NAT, retries once
- Credits exhausted? Check balance: [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)
- Burst rate limit? Wait a few minutes
- Run `status` in `./ask` for diagnostics

**macOS / Cloud-Only**
- Use `bash setup.sh --cloud` to skip GPU checks and model downloads
- `./ask` auto-selects the `ultrafast-nogpu` agent

## File Structure

```
.
├── setup.sh                          # One-time environment setup (--cloud for no-GPU path)
├── ask                               # Launch script (activates env, starts chat)
├── README.md                         # This file
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

## References

- [GAIA paper](https://arxiv.org/abs/2311.12983) (Mialon et al., 2023)
- [Student leaderboard](https://huggingface.co/spaces/agents-course/Students_Leaderboard): 20 Level-1 questions, scored
- [Official GAIA leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard): 300-question test set, answers hidden
- [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA): submission instructions, terms of use
- [NAT documentation](https://docs.nvidia.com/nemo/agent-toolkit/)
- [NAT source code and examples](https://github.com/NVIDIA/NeMo-Agent-Toolkit)

To compete on the official GAIA leaderboard, follow the submission instructions on the GAIA dataset page. Your agent works as-is; no code changes needed. The full 300-question test set takes 15-25 hours of GPU time per run.

---

**Tested with:** NAT 1.5.0, vLLM 0.18.0, Python 3.12, MiniMax M2.5 (local), Qwen 3.5-122B-A10B (cloud), on 8x H100 (Brev/GCP).
