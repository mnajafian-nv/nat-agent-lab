# Pre-Lab Setup: Agent Engineering with NeMo Agent Toolkit

**Please complete these steps before the lab session.** Setup takes 20-30 minutes and cannot be done during the lab itself.

You will build, debug, and benchmark LLM agents that use real tools (web search, code execution, vision, audio) to answer complex questions. The system runs on your 8xH100 GPU instance, and everything needs to be ready before the session starts.

---

## Step 1: Create your API keys

You need three free API keys. **Do this first** because account creation sometimes takes a few minutes.

1. **Tavily** (web search tool): Go to [tavily.com](https://tavily.com/) and sign in with your university/org Google account, or a personal Google account if you don't have one. Your API key appears on the dashboard. It starts with `tvly-`.

2. **NVIDIA Build** (vision and audio tools): Go to [build.nvidia.com](https://build.nvidia.com/) and create an account. Once logged in, visit [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys) and generate a key. It starts with `nvapi-`. Note: account creation requires phone verification. If you get stuck, email help@build.nvidia.com with your registered email and a screenshot.

3. **HuggingFace** (GAIA dataset and leaderboard): Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), create a token with **Read** access. It starts with `hf_`.

Save all three somewhere you can paste from. You will need them in Step 3.

## Step 2: Clone the repo on your GPU instance

SSH into your GPU instance and run:

```bash
git clone https://github.com/mnajafian-nv/nat-agent-lab.git nat-agent-lab
cd nat-agent-lab
```

## Step 3: Run setup

```bash
bash setup.sh
```

The script will:
- Install all dependencies
- Download the model (~220 GB, this is the slow part)
- Prompt you for the three API keys from Step 1

It runs inside tmux, so if your SSH connection drops, the setup keeps going. Just reconnect and run `tmux attach -t setup` to see progress.

## Step 4: Start the model server

```bash
bash gaia_tools/start_services.sh
```

This launches vLLM (serves the model) and Phoenix (tracing UI) in background sessions. Give vLLM a few minutes to load the model into GPU memory.

## Step 5: Verify

```bash
./ask
```

You should see a status line like:

```
  Agent: ultrafast | vLLM: OK | NAT: loading... | Phoenix: OK | Verbose: ON
```

Then a spinner (`Starting NAT |`) while the agent loads. Once it finishes, you see `Ready. Type a question, or 'help' for commands.` followed by the `ask>` prompt.

Type `What is 2+2?` and confirm you get a response. If the agent answers, you are ready for the lab.

---

## If something goes wrong

| Problem | Fix |
|---------|-----|
| Setup failed or SSH dropped mid-install | Run `bash setup.sh` again. It picks up where it left off. |
| vLLM shows "DOWN" after starting services | Wait 2-3 more minutes. The model is large. Check logs with `tmux attach -t vllm`. |
| API key rejected during setup | The script checks prefixes (`tvly-`, `nvapi-`, `hf_`). Double-check you copied the right value. |
| Need to change a key later | Edit `.env` in the repo root, or re-run `bash setup.sh`. |
| Reconnected via SSH, services still running | Just run `./ask` from the `nat-agent-lab` directory. |
| Services stopped after reconnect | Run `bash gaia_tools/start_services.sh` to restart them. |

If you are stuck after trying these, post in the class channel with the error message and someone will help you.

---

**Next step:** Once setup is verified, you are ready for the [Lab Guide](lab-guide.md).
