# NAT Agent Lab: Reference Guide

This guide covers everything beyond getting started. For setup and what to try first, see [README.md](../README.md).

---

## GAIA Questions

There are two separate question sets:

**Validation set** (165 questions with expected answers):

| Level | Count | Difficulty | Description |
|-------|-------|-----------|-------------|
| **1** | 53 | Easy | 1-2 tool calls, straightforward reasoning (e.g., web search + answer) |
| **2** | 86 | Medium | Multi-step reasoning, file handling, cross-referencing sources |
| **3** | 26 | Hard | Deep multi-tool chains, long reasoning, complex file analysis |

Use `level 1`, `level 2`, or `level 3` in `./ask` to list validation questions, then `level 1, 3` to run the 3rd question from Level 1. After each run the expected answer is shown so you can iterate.

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
| `gaia_results.json` | Every question with your agent's raw and cleaned answers (expected answers included only for validation set runs) |
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

Key details about this config:

- **`_type: openai`** uses ChatOpenAI pointed at NVIDIA Build's OpenAI-compatible endpoint. The `api_key` references `NGC_API_KEY` from your `.env` via environment variable interpolation.
- **Automatic retries.** NAT retries on 429, 500, 502, 503, 504 with exponential backoff.
- **Auto-detection.** When your YAML has a non-localhost `base_url`, the toolkit automatically skips vLLM checks and shows `vLLM: off (not needed)`.
- **Credits.** Cloud API calls consume credits. See [Notes on cloud usage](#notes-on-cloud-usage) for budget details.
- **Other models.** You can replace `model_name` with any model available at [build.nvidia.com](https://build.nvidia.com/), as long as it supports tool calling.

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

### Local vs. cloud: key differences

The `workflow:` and `functions:` sections stay the same regardless of which model you use. The `llms:` section is the only part that changes. Vision and audio tools (e.g., `describe_image`, `transcribe_audio`) always use the NVIDIA Build cloud API, independent of your main LLM choice.

| Parameter | Local (vLLM) | Cloud (NVIDIA Build) | Why it matters |
|-----------|-------------|---------------------|----------------|
| Model | MiniMax M2.5 (456B MoE, ~46B active) | Qwen 3.5-122B-A10B (MoE, 10B active) | Different reasoning capabilities |
| `_type` | `openai` (ChatOpenAI to local vLLM) | `openai` (ChatOpenAI to NVIDIA Build) | Same transport, different `base_url` |
| `max_tokens` | 16384 | **4096** (API hard limit) | Cloud responses truncate earlier; complex tool calls may get cut off |
| `frequency_penalty` | 0.3 | (not set) | NVIDIA Build ignores this parameter |
| `seed` | 42 | (not set) | NVIDIA Build ignores this parameter |
| Rate limit | None (local GPU) | ~40 RPM | NAT retries 429s with exponential backoff |
| Tool calling | Custom `minimax_m2` parser via vLLM | Native OpenAI-compatible tool calling | Different models, same protocol |

**Why Qwen instead of MiniMax on the cloud?** MiniMax uses a custom XML tool-call format that requires vLLM's `minimax_m2_tool_parser`. NVIDIA Build does not have this parser, so MiniMax tool calls fail on the cloud. Qwen 3.5 uses native OpenAI-compatible tool calling, which works directly with NVIDIA Build.

### Notes on cloud usage

- **NVIDIA Build rate limits.** Multi-step GAIA questions make 3-10 LLM calls; rapid consecutive calls can trigger 429 errors. The ultrafast-nogpu config uses `max_retries: 10` with exponential backoff, giving the rate limiter time to recover. Simple questions take ~30-60s; complex multi-step questions take ~2-3 minutes. If a question still fails with a 422, wait 60 seconds and retry. Back-to-back questions burn through the rate limit faster, so space them out during interactive use.
- **NVIDIA Build credit budget.** Free tier: 1,000 credits on signup (personal email); up to 5,000 with a business or institutional email (Profile > Request More). A 20-question benchmark uses ~500+ credits. With 5,000 credits you can run roughly 6-10 benchmarks plus interactive exploration. If credits run out, the toolkit detects persistent 429 errors and tells you to check your balance.
- **Reproducibility.** All local configs use `temperature: 0.0` and `seed: 42` for near-deterministic LLM output. The ultrafast-nogpu config uses `temperature: 0.0` but does not set `seed` (NVIDIA Build ignores this parameter for Qwen 3.5). Even with fixed seeds, answers can still vary between runs because: (1) tool results change over time (search results, Wikipedia edits), and (2) GPU floating-point arithmetic is non-deterministic across runs. This is inherent to agentic systems and is expected. If you need to compare runs, use the saved results in `<agent>/runs/` rather than re-running.
- **Custom agents are auto-detected.** When your custom YAML has a non-localhost `base_url` (e.g., `https://integrate.api.nvidia.com/v1`), the toolkit automatically skips vLLM checks and shows `vLLM: off (not needed)` in the status line. No special flags needed.
- **vLLM model loading.** vLLM serves one model per process. Running two large models simultaneously requires enough VRAM for both (MiniMax M2.5 alone uses ~220 GB).

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
- `./ask` auto-recovers: on a 422, it waits 30 s, restarts NAT, and retries the question once. If the retry also fails, check the items below.
- NVIDIA Build credits may be exhausted. Check your balance at: https://build.nvidia.com/settings/api-keys
- Request more credits with a business or institutional email (Profile > Request More)
- Wait a few minutes if you hit a burst rate limit
- Run `status` in `./ask` for a full API key diagnostic

**macOS / Cloud-Only Setup**
- Use `bash setup.sh --cloud` to skip GPU checks, vLLM, and model downloads entirely
- `./ask` auto-detects that vLLM is unavailable and starts with the `ultrafast-nogpu` agent
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

1. **Practice on the validation set** (already available via `level`): 165 questions across all 3 difficulty levels with expected answers.
2. **Submit to the official leaderboard**: Follow the submission instructions at the [GAIA dataset page](https://huggingface.co/datasets/gaia-benchmark/GAIA). The test set has 300 questions with no answers revealed; scoring is done server-side.
3. **Compare globally**: See how your agent ranks against published systems on the [official leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard).

Note: the full test set takes 15-25 hours of GPU time per agent run. Plan accordingly.
