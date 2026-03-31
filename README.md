# NAT Agent Lab

Hands-on lab for building, exploring, and benchmarking AI agents with [NVIDIA NeMo Agent Toolkit (NAT)](https://docs.nvidia.com/nemo/agent-toolkit/) on the [GAIA benchmark](https://arxiv.org/abs/2311.12983).

Clone the repo, run setup, type `./ask`, and start experimenting.

- **Four pre-built agent architectures** that solve the same problems with different engineering tradeoffs
- **Interactive REPL** with multi-turn memory and live tool tracing
- **GAIA benchmark** with automated [HuggingFace leaderboard](https://huggingface.co/spaces/agents-course/Students_Leaderboard) submission (20 scored questions)
- **165 dev questions** with expected answers for practice (Levels 1-3)
- **OpenTelemetry tracing** via Phoenix (visualize every tool call and LLM step)
- Works with a **local LLM via vLLM** (8x H100) or **NVIDIA Build cloud** (no GPU needed)
- 19 files, one entry point: `./ask`

The four included architectures all use the same tools:

| Agent | Config | Architecture | LLM |
|-------|--------|-------------|-----|
| **Single** | `single-agent/gaia_agent.yml` | `tool_calling_agent` with all tools | MiniMax M2.5 (local vLLM) |
| **Multi** | `multi-agent/gaia_agent_multi.yml` | `tool_calling_agent` orchestrator with 3 specialist sub-agents | MiniMax M2.5 (local vLLM) |
| **Ultrafast** | `ultrafast-agent/gaia_agent_ultrafast.yml` | `tool_calling_agent` with embedded routing in the system prompt | MiniMax M2.5 (local vLLM) |
| **Ultrafast-nogpu** | `ultrafast-nogpu-agent/gaia_agent_ultrafast_nogpu.yml` | Same ultrafast architecture, no local GPU needed | Qwen 3.5-122B-A10B (NVIDIA Build) |

## Prerequisites

Two setup paths are available. Pick whichever matches your hardware.

**Path A: Local LLM (8x H100)**

- Linux machine with **8x H100 GPUs** (or equivalent, ~640 GB VRAM total)
- **300 GB free disk space** for model weights and dependencies
- **NVIDIA drivers** installed (`nvidia-smi` works)

**Path B: Cloud LLM (any machine, no GPU needed for the LLM)**

- Any Linux machine (or macOS for local exploration)
- **NGC_API_KEY** with access to [build.nvidia.com](https://build.nvidia.com/)
- Uses Qwen 3.5-122B-A10B on NVIDIA Build (rate-limited to ~40 RPM, retries are automatic)
- **Credit limits:** 1,000 free credits on signup (personal email). Up to 5,000 with a business or institutional email (Profile > Request More). A full 20-question benchmark uses ~500+ credits. Check your balance at [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys).

Three API keys are needed (free tiers are sufficient):

- **Tavily**: [https://tavily.com/](https://tavily.com/) (internet search)
- **NVIDIA Build**: [https://build.nvidia.com/](https://build.nvidia.com/) (vision model API, or main LLM if not hosting locally)
- **HuggingFace**: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (GAIA dataset and leaderboard)

## Quick Start

### Path A: Local LLM (8x H100)

```bash
# 1. Clone the repo into a directory with 300+ GB free space
git clone <repo-url> nat-agent-lab
cd nat-agent-lab

# 2. Run setup (installs everything, downloads the model)
#    Takes ~20 minutes (mostly model download).
bash setup.sh

# 3. Start vLLM and Phoenix (in tmux background sessions)
#    Takes 5-10 minutes the first time (vLLM loads the model into GPU memory).
bash gaia_tools/start_services.sh

# 4. Start exploring!
./ask
```

You'll see a status line like `Agent: ultrafast | vLLM: OK | NAT: OK | Phoenix: OK | Verbose: ON`. If NAT shows OK, your setup is complete. Type `status` anytime for a full diagnostic including API key health.

### Path B: Cloud LLM (no local GPU)

```bash
# 1. Clone the repo
git clone <repo-url> nat-agent-lab
cd nat-agent-lab

# 2. Run setup with --cloud (skips vLLM install and model download)
bash setup.sh --cloud

# 3. (Optional) Start Phoenix for tracing
bash gaia_tools/start_services.sh --phoenix

# 4. Start exploring
./ask
# If vLLM is not running, the REPL auto-selects the ultrafast-nogpu agent.
```

You'll see a status line like `Agent: ultrafast-nogpu | vLLM: off (not needed) | NAT: OK | Phoenix: off | Verbose: ON`. If NAT shows OK, your setup is complete (Phoenix is optional). Type `status` anytime for a full diagnostic including API key health.

No vLLM needed. The REPL auto-detects that vLLM is not running and defaults to the ultrafast-nogpu agent, which runs on NVIDIA Build using your `NGC_API_KEY`.

**Coming back later?** If you've already run setup and services are still running, just do `./ask`. vLLM and Phoenix run in background tmux sessions that survive SSH disconnects. Each session starts fresh with the default agent (ultrafast if vLLM is running, ultrafast-nogpu otherwise). Agent switches, conversation history, and verbose settings are not persisted between sessions.

## Interactive Explorer (`./ask`)

The `./ask` launcher is the main entry point. It activates the environment, loads API keys, and starts the interactive REPL. On startup, it loads the **ultrafast** agent by default (or auto-selects **ultrafast-nogpu** if vLLM is not running). Everything you type (free-form questions, GAIA questions, follow-ups) goes through whichever agent is currently active. The status line always shows the active agent. Use `switch` to change agents, or `info` to see the active agent's model and tools.

Conversations are multi-turn: the agent remembers what you asked and can answer follow-ups. The prompt shows the turn count (e.g., `ask [3]>`). GAIA questions are always answered independently, then seeded into conversation so you can ask "why did you do that?" afterward.

**Commands:**

| Command | Description |
|---------|-------------|
| `<any text>` | Ask anything (multi-turn, remembers context) |
| `level l, n` | Run dev question n from GAIA level l (1,2,3) |
| `benchmark [agent]` | Run 20-question HF leaderboard, GAIA Level 1 (single, multi, ultrafast, ultrafast-nogpu, custom, or `all` for 3 local) |
| `switch [name]` | Change agent (single, multi, ultrafast, ultrafast-nogpu, or custom YAML) |
| `info` | Current agent, model, and tools |
| `status` | Services and API key status |
| `tracing` | View agent traces with Phoenix in your browser |
| `clear` | Reset conversation memory |
| `verbose on/off` | Show/hide full model reasoning |
| `help` | All commands |
| `quit` | Exit (resume with ./ask) |

## GAIA Questions

There are two separate question sets:

**Dev set** (165 questions with expected answers, for practice):

| Level | Count | Difficulty | Description |
|-------|-------|-----------|-------------|
| **1** | 53 | Easy | 1-2 tool calls, straightforward reasoning (e.g., web search + answer) |
| **2** | 86 | Medium | Multi-step reasoning, file handling, cross-referencing sources |
| **3** | 26 | Hard | Deep multi-tool chains, long reasoning, complex file analysis |

Use `level 1`, `level 2`, or `level 3` in the REPL to browse dev questions, then `level 1, 3` to run question 3 from Level 1. After each run the expected answer is shown so you can iterate.

**Leaderboard set** (20 scored questions, no answers revealed):

The `benchmark` command fetches these directly from the [HF course scoring API](https://huggingface.co/spaces/agents-course/Students_Leaderboard), runs your agent, and submits answers for scoring. Your score is based only on these 20 questions.

## Running the Full Benchmark

The leaderboard benchmark fetches 20 questions from the HF course API, runs your agent on each, and submits answers for scoring. Use the `benchmark` command inside `./ask`, or run directly from the command line:

```bash
# Single agent (local vLLM)
bash gaia_tools/gaia_run.sh --single

# Multi-agent (local vLLM)
bash gaia_tools/gaia_run.sh --multi

# Ultrafast agent (local vLLM)
bash gaia_tools/gaia_run.sh --ultrafast

# ultrafast-nogpu agent (NVIDIA Build, no local GPU)
bash gaia_tools/gaia_run.sh --ultrafast-nogpu

# Custom agent config
bash gaia_tools/gaia_run.sh -c my-agent/gaia_agent.yml

# All three local agents sequentially
bash gaia_tools/gaia_run_all.sh
```

Each run prompts for an org name and team name, and auto-increments a run counter. Your leaderboard entry appears as `NAT-<org>-<team>-<agent>` (e.g., `NAT-UCB-AgentSmiths-SingleAgent`).

### Where to find your results

After each benchmark run, results are saved locally and submitted to the public leaderboard:

**Public score:** Open the [HF course leaderboard](https://huggingface.co/spaces/agents-course/Students_Leaderboard) and search for your team name. Your score appears within seconds of submission.

**Local results directory:** Each run is saved to `<agent>/runs/runN_<config>/` (auto-numbered). For example, `ultrafast-agent/runs/run3_gaia_agent_ultrafast/` contains:

| File | What it contains |
|------|-----------------|
| `gaia_summary.json` | HF leaderboard score and correct count, time per question, per-level breakdown |
| `gaia_results.json` | Every question with your agent's raw and cleaned answers (expected answers included only for dev set runs) |
| `benchmark.log` | Full terminal output (what you saw on screen) |
| `nat.log` | NAT server log (useful for debugging tool call failures) |
| `config.yml` | Snapshot of the YAML config used for this run |

The most recent run is always symlinked at `<agent>/runs/latest/`, so you can quickly check your last score:

```bash
cat ultrafast-agent/runs/latest/gaia_summary.json
```

To see a history of all runs and their scores:

```bash
bash gaia_tools/gaia_run.sh --history
```

## Phoenix Tracing

Phoenix is an OpenTelemetry-based tracing UI that visualizes every tool call, LLM prompt, and agent step. No account or API key is needed (self-hosted).

Phoenix starts automatically with `start_services.sh`. To view traces, open [http://localhost:6006](http://localhost:6006) in your browser.

**If you are SSH'd into a remote machine (Brev, GCP, etc.)**, you need to forward port 6006 to your laptop first. Open a **separate terminal on your laptop** (not on the remote machine) and run:

```bash
ssh -L 6006:localhost:6006 <your-ssh-host>
```

Replace `<your-ssh-host>` with whatever you normally use to SSH in (e.g., `ssh my-gpu-box`). Keep this terminal open while you use Phoenix, then open [http://localhost:6006](http://localhost:6006) in your browser.

**If you have a browser on the same machine** (local GPU workstation with a desktop), just open [http://localhost:6006](http://localhost:6006) directly. No port forwarding needed.

All four agent configs have Phoenix tracing pre-configured. Traces appear automatically when Phoenix is running. If Phoenix is down, agents still work normally.

## Building Your Own Agent

1. Copy one of the existing configs as a starting point:
   ```bash
   mkdir my-agent
   cp single-agent/gaia_agent.yml my-agent/gaia_agent.yml
   ```

2. Edit the YAML to change the system prompt, tools, agent type, or model settings.

3. Test interactively:
   ```bash
   ./ask
   # Then: switch my-agent/gaia_agent.yml
   ```

4. Run the benchmark with your config:
   ```bash
   bash gaia_tools/gaia_run.sh -c my-agent/gaia_agent.yml
   ```

## Using a Different Model

The included agents use MiniMax M2.5 served locally via vLLM. You can swap to a different model without changing any toolkit code. Just edit the `llms:` section in your YAML config.

### Option A: NVIDIA Build (no local GPU needed)

The included `ultrafast-nogpu-agent/gaia_agent_ultrafast_nogpu.yml` already uses Qwen 3.5-122B-A10B on NVIDIA Build. To use a different model, edit the `llms:` block:

```yaml
llms:
  nim_llm:
    _type: openai
    base_url: "https://integrate.api.nvidia.com/v1"
    api_key: ${NGC_API_KEY}
    model_name: qwen/qwen3.5-122b-a10b   # any model on build.nvidia.com
    temperature: 0.0
    max_tokens: 4096
```

This uses `_type: openai` (ChatOpenAI) pointed at NVIDIA Build's OpenAI-compatible API endpoint. The `api_key` references `NGC_API_KEY` from your `.env` file via environment variable interpolation. NAT retries on 429, 500, 502, 503, 504 with exponential backoff automatically. Why not MiniMax M2.5 on the cloud? MiniMax uses a custom XML tool-call format that requires the `minimax_m2_tool_parser` in vLLM. NVIDIA Build does not have this parser, so MiniMax tool calls break on the cloud. Qwen 3.5-122B-A10B uses a Mixture-of-Experts architecture (122B total, 10B active per token) with native tool calling on NVIDIA Build, delivering fast inference with strong reasoning. Browse available models at [build.nvidia.com](https://build.nvidia.com/). Cloud API calls consume credits from your NVIDIA Build account (1,000 free on signup, up to 5,000 with a business or institutional email). The toolkit auto-detects non-localhost `base_url` values and skips vLLM checks automatically.

### Option B: Different local model via vLLM

1. Stop the current vLLM instance:
   ```bash
   bash gaia_tools/start_services.sh --stop
   ```

2. Start vLLM with your model (adjust flags for your model's requirements):
   ```bash
   vllm serve <model-id> --trust-remote-code \
       --tensor-parallel-size 8 \
       --enable-auto-tool-choice --tool-call-parser hermes \
       --port 9000
   ```
   The `--tool-call-parser` flag depends on your model (e.g., `hermes` for Hermes-based models, `llama3_json` for Llama 3, `minimax_m2` for MiniMax). Check [vLLM's supported models](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#tool-calling-in-the-chat-completion-api).

3. Update your YAML to match:
   ```yaml
   llms:
     nim_llm:
       _type: openai
       base_url: "http://localhost:9000/v1"
       model_name: <model-id>              # must match what vLLM is serving
       api_key: "not-needed"
       temperature: 0.6
       max_tokens: 8192
   ```

4. Test it:
   ```bash
   ./ask
   # Then: switch my-agent/gaia_agent.yml
   ```

### Notes

- The `workflow:` and `functions:` sections stay the same regardless of which model you use. The `llms:` section is the only part that changes.
- Vision and audio tools (e.g., `describe_image`, `transcribe_audio`) always use the NVIDIA Build cloud API, independent of your main LLM choice.
- vLLM serves one model per process. Running two large models simultaneously requires enough VRAM for both (MiniMax M2.5 alone uses ~220 GB).
- **Custom agents are auto-detected.** When your custom YAML has a non-localhost `base_url` (e.g., `https://integrate.api.nvidia.com/v1`), the toolkit automatically skips vLLM checks and shows `vLLM: off (not needed)` in the status line. No special flags needed.
- **Local vs ultrafast-nogpu: key differences.** The ultrafast-nogpu agent uses a different model with different API limits. Understanding these tradeoffs is critical for interpreting benchmark results.

  | Parameter | Local (vLLM) | Cloud (NVIDIA Build) | Why it matters |
  |-----------|-------------|---------------------|----------------|
  | Model | MiniMax M2.5 (456B MoE, ~46B active) | Qwen 3.5-122B-A10B (MoE, 10B active) | Different reasoning capabilities |
  | `_type` | `openai` (ChatOpenAI to local vLLM) | `openai` (ChatOpenAI to NVIDIA Build) | Same transport, different `base_url` |
  | `max_tokens` | 16384 | **4096** (API hard limit) | Cloud responses truncate earlier; complex tool calls may get cut off |
  | `frequency_penalty` | 0.3 | (not set) | NVIDIA Build ignores this parameter |
  | `seed` | 42 | (not set) | NVIDIA Build ignores this parameter |
  | Rate limit | None (local GPU) | ~40 RPM | NAT retries 429s with exponential backoff |
  | Tool calling | Custom `minimax_m2` parser via vLLM | Native OpenAI-compatible tool calling | Different models, same protocol |

  Why Qwen 3.5-122B-A10B instead of MiniMax M2.5 on the cloud? MiniMax uses a custom XML tool-call format that requires vLLM's `minimax_m2_tool_parser`. NVIDIA Build does not have this parser. Qwen 3.5 is a Mixture-of-Experts model (122B total, 10B active per token) with native OpenAI-compatible tool calling, delivering fast inference with strong reasoning on NVIDIA Build.

- **NVIDIA Build rate limits.** Multi-step GAIA questions make 3-10 LLM calls; rapid consecutive calls can trigger 429 errors. The ultrafast-nogpu config uses `max_retries: 10` with exponential backoff, giving the rate limiter time to recover. Simple questions take ~30-60s; complex multi-step questions take ~2-3 minutes. If a question still fails with a 422, wait 60 seconds and retry. Back-to-back questions burn through the rate limit faster, so space them out during interactive use.
- **NVIDIA Build credit budget.** Free tier: 1,000 credits on signup (personal email); up to 5,000 with a business or institutional email (Profile > Request More). A 20-question benchmark uses ~500+ credits. With 5,000 credits you can run roughly 6-10 benchmarks plus interactive exploration. If credits run out, the toolkit detects persistent 429 errors and tells you to check your balance.
- **Reproducibility.** All local configs use `temperature: 0.0` and `seed: 42` for near-deterministic LLM output. The ultrafast-nogpu config uses `temperature: 0.0` but does not set `seed` (NVIDIA Build ignores this parameter for Qwen 3.5). Even with fixed seeds, answers can still vary between runs because: (1) tool results change over time (search results, Wikipedia edits), and (2) GPU floating-point arithmetic is non-deterministic across runs. This is inherent to agentic systems and is expected. If you need to compare runs, use the saved results in `<agent>/runs/` rather than re-running.

## File Structure

After cloning and running `setup.sh`, your directory looks like this:

```
.
├── setup.sh                          # One-time environment setup (--cloud for no-GPU path)
├── ask                               # Launch script (activates env, runs REPL)
├── README.md                         # This file
├── LICENSE                           # Apache 2.0 (matches upstream NAT)
├── .env.example                      # API key template
├── gaia_questions.json               # GAIA questions, all levels (downloaded by setup.sh)
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
    ├── ask.py                        # Interactive REPL (called by ./ask)
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

## Services

Three background services run while you work:

| Service | Port | tmux session | What it does |
|---------|------|-------------|-------------|
| vLLM | 9000 | `vllm` | Serves the LLM (slow to restart, leave it running). Not needed for ultrafast-nogpu. |
| Phoenix | 6006 | `phoenix` | Tracing UI (optional, negligible resources) |
| NAT | 8000 | (managed by `./ask`) | Agent runtime (restarted on `switch`) |

```bash
# Check status of all services
bash gaia_tools/start_services.sh --check

# Restart everything (if vLLM crashed or machine rebooted)
bash gaia_tools/start_services.sh

# Stop everything (frees GPU memory)
bash gaia_tools/start_services.sh --stop
```

From inside `./ask`, type `status` to check services.

## Troubleshooting

**vLLM won't start or crashes**
- Check GPU memory: `nvidia-smi`
- Kill stale processes: `bash gaia_tools/start_services.sh --stop`
- Verify model downloaded: `ls .cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.5/`

**NAT fails to start**
- Check config syntax: `bash gaia_tools/gaia_run.sh --dry-run --single`
- Check port 8000: `curl localhost:8000/health`
- View NAT log: look in the run directory or `/tmp/nat_serve/`

**Phoenix not accessible from laptop**
- Verify Phoenix is running: `curl localhost:6006`
- Check port forwarding is active
- Phoenix is optional; agents work without it

**"Disk space insufficient" during setup**
- Clone the repo to a partition with 300+ GB free (e.g., `/ephemeral/`, `/data/`, `/mnt/`)
- Or run `bash setup.sh --cloud` to skip the local model entirely
- Check space with: `df -h .`

**`./ask` says vLLM is DOWN after logging back in**
- vLLM runs in a tmux session. If the machine rebooted, restart it: `bash gaia_tools/start_services.sh`
- Check if vLLM is still loading: `tmux attach -t vllm` (Ctrl+B then D to detach)

**Questions show "[file not found]"**
- Ensure `gaia_files/` directory exists with attached files
- Re-run `python3 gaia_tools/prep_gaia_data.py` to re-download them

**422 "Unprocessable Entity" errors (ultrafast-nogpu)**
- The REPL auto-recovers: on a 422, it waits 30 s, restarts NAT, and retries the question once. If the retry also fails, check the items below.
- NVIDIA Build credits may be exhausted. Check your balance at: https://build.nvidia.com/settings/api-keys
- Request more credits with a business or institutional email (Profile > Request More)
- Wait a few minutes if you hit a burst rate limit
- Run `status` in the REPL for a full API key diagnostic

**macOS / Cloud-Only Setup**
- Use `bash setup.sh --cloud` to skip GPU checks, vLLM, and model downloads entirely
- The REPL auto-detects that vLLM is unavailable and starts with the `ultrafast-nogpu` agent
- If `prep_gaia_data.py` fails with a permission error on the HuggingFace cache, re-run `bash setup.sh --cloud` (it now auto-sets a local `HF_HOME`)
- Port forwarding for Phoenix: `ssh -L 6006:localhost:6006 <your-server>` (or use VS Code remote)

## Going Further

**Useful links:**
- [GAIA paper](https://arxiv.org/abs/2311.12983) (Mialon et al., 2023): the benchmark design, methodology, and baseline results
- [HF course leaderboard](https://huggingface.co/spaces/agents-course/Students_Leaderboard): where your `benchmark` scores appear (20 questions)
- [Official GAIA leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard): the full research benchmark (300 questions)
- [GAIA dataset on HuggingFace](https://huggingface.co/datasets/gaia-benchmark/GAIA): dataset card, submission instructions, terms of use

**Compete on the official GAIA leaderboard:**

The `benchmark` command runs 20 questions from the HF course API. If you want to compete on the official GAIA leaderboard (used by research teams worldwide), you can submit your agent's results to the full 300-question test set.

Your agent, tools, and prompts work as-is. No code changes needed.

1. **Practice on the dev set** (already available via `level`): 165 questions across all 3 difficulty levels with expected answers.
2. **Submit to the official leaderboard**: Follow the submission instructions at the [GAIA dataset page](https://huggingface.co/datasets/gaia-benchmark/GAIA). The test set has 300 questions with no answers revealed; scoring is done server-side.
3. **Compare globally**: See how your agent ranks against published systems on the [official leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard).

Note: the full test set takes 15-25 hours of GPU time per agent run. Plan accordingly.
