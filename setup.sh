#!/usr/bin/env bash
# ============================================================================
# setup.sh - One-time environment setup for GAIA Agent Toolkit
#
# Installs all dependencies, downloads the model, and verifies the environment.
# Safe to re-run (all steps are idempotent).
#
# Path A (GPU):    Linux with 8x H100 GPUs, vLLM + MiniMax M2.5
#                  Requires: 300 GB free disk, NVIDIA driver (nvidia-smi works)
# Path B (Ollama): macOS or Linux, no GPU, Ollama + Qwen3.5 35B-A3B
#                  Requires: 32+ GB RAM
#
# Usage:
#   cd <repo-root>
#   bash setup.sh            # auto-detects GPU or Ollama path
# ============================================================================
set -uo pipefail

# Resolve absolute path before tmux wrapper (tmux may start in a different cwd)
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

# ---- Auto-wrap in tmux to survive SSH disconnects ----
if [ -z "${TMUX:-}" ] && command -v tmux &>/dev/null; then
    tmux kill-session -t setup 2>/dev/null || true
    echo "  Starting setup in tmux session 'setup' (survives SSH disconnects)..."
    exec tmux new-session -s setup "bash \"$SCRIPT_PATH\" $*; echo; echo 'Setup complete. Press Enter to close.'; read"
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }
die()  { err "$*"; exit 1; }

# Strip leading/trailing whitespace and surrounding quotes from a pasted value.
clean_key() {
    local val="$1"
    val="${val#"${val%%[![:space:]]*}"}"   # trim leading whitespace
    val="${val%"${val##*[![:space:]]}"}"   # trim trailing whitespace
    val="${val#\"}" ; val="${val%\"}"       # strip surrounding double quotes
    val="${val#\'}" ; val="${val%\'}"       # strip surrounding single quotes
    echo "$val"
}

# ============================================================================
# Detect setup path: GPU (vLLM + MiniMax M2.5) or Ollama (no GPU)
# ============================================================================
if nvidia-smi &>/dev/null; then
    MODE=gpu
    echo ""
    echo "============================================================"
    echo "  NAT Agent Lab - Environment Setup (Path A: GPU / vLLM)"
    echo "============================================================"
    echo "  Repo: $REPO_ROOT"
    echo ""
else
    MODE=ollama
    echo ""
    echo "============================================================"
    echo "  NAT Agent Lab - Environment Setup (Path B: Ollama / no GPU)"
    echo "============================================================"
    echo "  Repo: $REPO_ROOT"
    echo "  No GPU detected. Using Ollama path (Qwen3.5, local inference)."
    echo ""
fi

# ============================================================================
# Step 1: Disk space (GPU) or RAM check (Ollama)
# ============================================================================
if [ "$MODE" = gpu ]; then
    log "Step 1/8: Checking disk space..."
    AVAIL_GB=$(df -BG "$REPO_ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
    if [ "$AVAIL_GB" -lt 300 ]; then
        die "Only ${AVAIL_GB}GB free. Need at least 300GB for model weights.
  Clone the repo to a directory with more space (e.g., /ephemeral/, /data/)."
    fi
    ok "Disk space: ${AVAIL_GB}GB available"
else
    log "Step 1/8: Checking RAM for Ollama model selection..."
    if [[ "$(uname)" == "Darwin" ]]; then
        RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        RAM_GB=$(( RAM_BYTES / 1024 / 1024 / 1024 ))
    else
        RAM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
        RAM_GB=$(( RAM_KB / 1024 / 1024 ))
    fi
    if [ "$RAM_GB" -ge 32 ]; then
        ok "RAM: ${RAM_GB}GB, using qwen3.5:35b-a3b (~24 GB, best accuracy)"
        OLLAMA_MODEL="qwen3.5:35b-a3b"
    else
        die "RAM: ${RAM_GB}GB. qwen3.5:35b-a3b requires 32+ GB RAM. Upgrade your machine or use the GPU path."
    fi
fi

# ============================================================================
# Step 2: Python and virtual environment
# ============================================================================
log "Step 2/8: Setting up Python environment..."

if command -v uv &>/dev/null; then
    ok "uv already installed"
else
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv installed"
fi

if ! command -v python3.12 &>/dev/null && command -v uv &>/dev/null; then
    log "Installing Python 3.12 via uv..."
    uv python install 3.12
fi

if [ -f ".venv/bin/activate" ]; then
    ok "Virtual environment already exists"
else
    log "Creating virtual environment..."
    uv venv --python 3.12 .venv 2>/dev/null || python3 -m venv .venv
fi

if [ ! -f ".venv/bin/activate" ]; then
    die "Virtual environment not created. Check Python installation and disk space."
fi
source .venv/bin/activate
ok "Python $(python3 --version) in .venv"

# ============================================================================
# Step 3: Install Python packages
# ============================================================================
log "Step 3/8: Installing Python packages..."

if python3 -c "import nat; import datasets; import openpyxl; import bs4; import pypdf; import pptx; import sympy; import dask; import distributed" 2>/dev/null; then
    ok "NAT and dependencies already installed"
else
    log "Installing NAT and dependencies..."
    uv pip install "nvidia-nat[langchain,phoenix]==1.5.0" "arize-phoenix==13.21.0" "arize-phoenix-evals>=2.12.0,<3" requests pyyaml datasets \
        openpyxl beautifulsoup4 pypdf python-pptx sympy dask distributed 2>/dev/null \
        || pip install "nvidia-nat[langchain,phoenix]==1.5.0" "arize-phoenix==13.21.0" "arize-phoenix-evals>=2.12.0,<3" requests pyyaml datasets \
        openpyxl beautifulsoup4 pypdf python-pptx sympy dask distributed
    ok "NAT installed"
fi

# Ensure arize-phoenix-evals is <3 (3.0+ removed phoenix.evals.models used by arize-phoenix==13.21.0)
if python3 -c "from phoenix.evals.models.rate_limiters import RateLimiter" 2>/dev/null; then
    ok "arize-phoenix-evals version OK"
else
    log "Downgrading arize-phoenix-evals to 2.x (3.0+ is incompatible with arize-phoenix==13.21.0)..."
    uv pip install "arize-phoenix-evals>=2.12.0,<3" 2>/dev/null \
        || pip install "arize-phoenix-evals>=2.12.0,<3"
    ok "arize-phoenix-evals downgraded"
fi

if python3 -c "from gaia_tools.register import read_file" 2>/dev/null; then
    ok "GAIA custom tools already installed"
elif [ -f "gaia_tools/pyproject.toml" ]; then
    log "Installing GAIA custom tools (read_file, fetch_url, python_executor, etc.)..."
    uv pip install -e gaia_tools/ 2>/dev/null || pip install -e gaia_tools/
    ok "GAIA tools installed"
fi

if [ "$MODE" = gpu ]; then
    if python3 -c "import vllm" 2>/dev/null; then
        ok "vLLM already installed"
    else
        log "Installing vLLM (this may take a few minutes)..."
        uv pip install "vllm==0.18.0" --torch-backend=auto 2>/dev/null \
            || pip install "vllm==0.18.0"
        ok "vLLM installed"
    fi
else
    ok "Skipping vLLM (not needed for Ollama path)"
fi

# ============================================================================
# Step 4: System dependencies
# ============================================================================
log "Step 4/8: Checking system dependencies..."

if command -v stockfish &>/dev/null; then
    ok "Stockfish already installed"
elif command -v apt-get &>/dev/null; then
    log "Installing Stockfish chess engine..."
    sudo apt-get update -qq && sudo apt-get install -y -qq stockfish 2>/dev/null \
        && ok "Stockfish installed" \
        || warn "Stockfish install failed (solve_chess will use fallback)"
elif command -v brew &>/dev/null; then
    log "Installing Stockfish chess engine via Homebrew..."
    brew install stockfish 2>/dev/null \
        && ok "Stockfish installed" \
        || warn "Stockfish install failed (solve_chess will use fallback)"
else
    warn "Stockfish not available (install manually: brew install stockfish or apt-get install stockfish)"
fi

# ============================================================================
# Step 5: API keys
# ============================================================================
log "Step 5/8: Checking API keys..."

if [ -f ".env" ]; then
    if (set -a; source .env; set +a) 2>/dev/null; then
        set -a
        source .env
        set +a
        # Clean values in case .env was hand-edited with extra quotes or spaces
        [ -n "${TAVILY_API_KEY:-}" ] && TAVILY_API_KEY="$(clean_key "$TAVILY_API_KEY")"
        [ -n "${NGC_API_KEY:-}" ]    && NGC_API_KEY="$(clean_key "$NGC_API_KEY")"
        [ -n "${HF_TOKEN:-}" ]       && HF_TOKEN="$(clean_key "$HF_TOKEN")"
        ok ".env file found and loaded"
    else
        warn ".env file has syntax errors. Ignoring it and prompting for keys."
        warn "  (The broken .env will be overwritten with correct values.)"
    fi
fi

while true; do
    if [ -z "${TAVILY_API_KEY:-}" ]; then
        echo ""
        echo "  Tavily API key is needed for the internet search tool."
        echo "  1. Go to https://tavily.com/ and sign in with Google (or use your university/org account)."
        echo "  2. Your API key is on the dashboard after signing in. Copy it."
        read -rp "  TAVILY_API_KEY: " TAVILY_API_KEY
        TAVILY_API_KEY="$(clean_key "$TAVILY_API_KEY")"
        [ -z "$TAVILY_API_KEY" ] && die "Tavily key is required."
    fi
    if [[ "$TAVILY_API_KEY" =~ ^tvly- ]]; then
        break
    fi
    err "TAVILY_API_KEY must start with 'tvly-'. You entered: ${TAVILY_API_KEY:0:20}..."
    echo "  Make sure you copied the API key from your Tavily dashboard, not something else."
    TAVILY_API_KEY=""
done
ok "TAVILY_API_KEY set (${TAVILY_API_KEY:0:12}...)"

if [ "$MODE" = gpu ]; then
    while true; do
        if [ -z "${NGC_API_KEY:-}" ]; then
            echo ""
            echo "  NVIDIA Build API key is needed for vision and audio tools."
            echo "  1. Go to https://build.nvidia.com/ and click Login (top right)."
            echo "  2. Create an NVIDIA account if you don't have one (verify by email and phone)."
            echo "  3. Once logged in, go to https://build.nvidia.com/settings/api-keys"
            echo "     and click Generate API Key. Copy it."
            read -rp "  NGC_API_KEY: " NGC_API_KEY
            NGC_API_KEY="$(clean_key "$NGC_API_KEY")"
            [ -z "$NGC_API_KEY" ] && die "NGC key is required."
        fi
        if [[ "$NGC_API_KEY" =~ ^nvapi- ]]; then
            break
        fi
        err "NGC_API_KEY must start with 'nvapi-'. You entered: ${NGC_API_KEY:0:20}..."
        echo "  Make sure you copied the API Key from https://build.nvidia.com/settings/api-keys"
        NGC_API_KEY=""
    done
    ok "NGC_API_KEY set (${NGC_API_KEY:0:12}...)"
else
    # Ollama path: NGC is optional (only needed for describe_image / transcribe_audio)
    if [ -z "${NGC_API_KEY:-}" ]; then
        echo ""
        echo "  NVIDIA Build API key is optional for the Ollama path."
        echo "  It enables vision (describe_image) and audio (transcribe_audio) tools."
        echo "  1. Go to https://build.nvidia.com/ and click Login (top right)."
        echo "  2. Create an NVIDIA account if needed, then: profile icon > API Keys > Generate."
        echo "  Press Enter to skip if you don't need vision/audio tools."
        read -rp "  NGC_API_KEY (optional): " NGC_API_KEY
        NGC_API_KEY="$(clean_key "$NGC_API_KEY")"
        if [ -n "$NGC_API_KEY" ]; then
            if [[ ! "$NGC_API_KEY" =~ ^nvapi- ]]; then
                err "NGC_API_KEY must start with 'nvapi-'. You entered: ${NGC_API_KEY:0:20}..."
                echo "  Ignoring invalid key. describe_image and transcribe_audio will not work."
                NGC_API_KEY=""
            else
                ok "NGC_API_KEY set (${NGC_API_KEY:0:12}...)"
            fi
        else
            warn "NGC_API_KEY skipped. describe_image and transcribe_audio will not work."
        fi
    else
        if [[ ! "$NGC_API_KEY" =~ ^nvapi- ]]; then
            warn "NGC_API_KEY in .env doesn't start with 'nvapi-'. Vision/audio tools may not work."
        fi
        ok "NGC_API_KEY set (${NGC_API_KEY:0:12}...)"
    fi
fi

while true; do
    if [ -z "${HF_TOKEN:-}" ]; then
        echo ""
        echo "  HuggingFace token is needed for the GAIA dataset and leaderboard submission."
        echo "  1. Go to https://huggingface.co/ and sign in (or create a free account)."
        echo "  2. Go to Settings > Access Tokens: https://huggingface.co/settings/tokens"
        echo "  3. Create a token with Read access. Copy it."
        read -rp "  HF_TOKEN: " HF_TOKEN
        HF_TOKEN="$(clean_key "$HF_TOKEN")"
        [ -z "$HF_TOKEN" ] && die "HuggingFace token is required."
    fi
    if [[ "$HF_TOKEN" =~ ^hf_ ]]; then
        break
    fi
    err "HF_TOKEN must start with 'hf_'. You entered: ${HF_TOKEN:0:20}..."
    echo "  Make sure you copied the token from https://huggingface.co/settings/tokens"
    HF_TOKEN=""
done
ok "HF_TOKEN set (${HF_TOKEN:0:8}...)"

# Always write .env so keys survive across SSH sessions.
# Pass keys via env vars (not string interpolation) to avoid issues with special chars.
export TAVILY_API_KEY HF_TOKEN
[ -n "${NGC_API_KEY:-}" ] && export NGC_API_KEY
export _SETUP_HF_HOME="$REPO_ROOT/.cache/huggingface"

python3 -c "
import os, re

keys_to_write = {
    'TAVILY_API_KEY': os.environ.get('TAVILY_API_KEY', ''),
    'NGC_API_KEY': os.environ.get('NGC_API_KEY', ''),
    'HF_TOKEN': os.environ.get('HF_TOKEN', ''),
    'HF_HOME': os.environ.get('_SETUP_HF_HOME', ''),
}

env_path = '.env'
lines = open(env_path).readlines() if os.path.exists(env_path) else []

for key, val in keys_to_write.items():
    if not val:
        continue
    pattern = re.compile(r'^(export\s+)?' + re.escape(key) + r'=')
    found = False
    for i, line in enumerate(lines):
        if pattern.match(line.strip()):
            lines[i] = f\"{key}='{val}'\n\"
            found = True
            break
    if not found:
        lines.append(f\"{key}='{val}'\n\")

with open(env_path, 'w') as f:
    f.writelines(lines)
"
unset _SETUP_HF_HOME

if [ -f ".env" ]; then
    ok "Keys saved to .env"
else
    die "Failed to write .env. Check file permissions in $REPO_ROOT"
fi

# ============================================================================
# Step 6: Model setup
# ============================================================================
if [ "$MODE" = gpu ]; then
    log "Step 6/8: Checking model weights..."

    export HF_HOME="$REPO_ROOT/.cache/huggingface"
    mkdir -p "$HF_HOME"

    MODEL_ID="MiniMaxAI/MiniMax-M2.5"
    MODEL_CACHE="$HF_HOME/hub/models--MiniMaxAI--MiniMax-M2.5"

    if [ -d "$MODEL_CACHE" ]; then
        SHARD_COUNT=$(find "$MODEL_CACHE" -name "*.safetensors" 2>/dev/null | wc -l)
        if [ "$SHARD_COUNT" -ge 40 ]; then
            ok "Model already downloaded ($SHARD_COUNT shards)"
        else
            warn "Model partially downloaded ($SHARD_COUNT shards). Resuming..."
            huggingface-cli download "$MODEL_ID"
            ok "Model download complete"
        fi
    else
        log "Downloading $MODEL_ID (~220GB). This takes 15-30 minutes..."
        log "  Progress will show below. Safe to disconnect SSH (running in tmux)."
        echo ""
        huggingface-cli download "$MODEL_ID"
        ok "Model download complete"
    fi
else
    log "Step 6/8: Setting up Ollama and pulling model..."

    # Install Ollama if not present
    if command -v ollama &>/dev/null; then
        ok "Ollama already installed ($(ollama --version 2>/dev/null || echo 'version unknown'))"
    else
        log "Installing Ollama..."
        if [[ "$(uname)" == "Darwin" ]]; then
            if command -v brew &>/dev/null; then
                brew install ollama \
                    && ok "Ollama installed via Homebrew" \
                    || die "Ollama install failed. Install manually: https://ollama.com/"
            else
                die "Homebrew not found. Install Ollama manually from https://ollama.com/ then re-run setup."
            fi
        else
            curl -fsSL https://ollama.com/install.sh | sh \
                && ok "Ollama installed" \
                || die "Ollama install failed. Install manually: https://ollama.com/"
        fi
    fi

    # Start ollama serve if not already running
    if curl -sf http://localhost:11434 &>/dev/null; then
        ok "Ollama already running on port 11434"
    else
        log "Starting Ollama server..."
        if [[ "$(uname)" == "Darwin" ]] && command -v brew &>/dev/null \
                && brew services list 2>/dev/null | grep -q ollama; then
            brew services start ollama 2>/dev/null || true
        else
            nohup ollama serve > /tmp/ollama_serve.log 2>&1 &
            disown
        fi
        # Wait for Ollama to be ready
        WAITED=0
        while ! curl -sf http://localhost:11434 &>/dev/null; do
            sleep 2
            WAITED=$((WAITED + 2))
            if [ $WAITED -ge 30 ]; then
                die "Ollama server didn't start after 30s. Check /tmp/ollama_serve.log"
            fi
            printf "."
        done
        echo ""
        ok "Ollama server ready after ${WAITED}s"
    fi

    # Pull the model
    if ollama list 2>/dev/null | grep -q "^${OLLAMA_MODEL}"; then
        ok "Model ${OLLAMA_MODEL} already pulled"
    else
        log "Pulling ${OLLAMA_MODEL} (this may take several minutes)..."
        ollama pull "${OLLAMA_MODEL}" \
            && ok "Model ${OLLAMA_MODEL} ready" \
            || die "Failed to pull ${OLLAMA_MODEL}. Check your internet connection."
    fi

    # Update the Ollama agent config to match the selected model
    OLLAMA_CONFIG="ultrafast-ollama-agent/gaia_agent_ultrafast_ollama.yml"
    if [ -f "$OLLAMA_CONFIG" ]; then
        python3 -c "
import re, sys
path = '$OLLAMA_CONFIG'
model = '$OLLAMA_MODEL'
text = open(path).read()
updated = re.sub(r'(model_name:\s*)qwen3\.5:[\w-]+', r'\g<1>' + model, text)
if updated != text:
    open(path, 'w').write(updated)
    print('  Updated model_name to ' + model + ' in ' + path)
"
    fi
fi

# ============================================================================
# Step 7: GAIA questions and files
# ============================================================================
# Ensure HF cache is local and writable (.env may have a stale HF_HOME
# from another machine, e.g. /ephemeral on a GPU instance).
if [ -z "${HF_HOME:-}" ] || ! mkdir -p "$HF_HOME" 2>/dev/null; then
    export HF_HOME="$REPO_ROOT/.cache/huggingface"
    mkdir -p "$HF_HOME"
fi

log "Step 7/8: Checking GAIA questions..."

if [ -f "gaia_questions.json" ] && [ -f "gaia_dev_questions.json" ]; then
    Q_COUNT=$(python3 -c "import json; print(len(json.load(open('gaia_questions.json'))))" 2>/dev/null || echo "0")
    DEV_COUNT=$(python3 -c "import json; print(len(json.load(open('gaia_dev_questions.json'))))" 2>/dev/null || echo "0")
    ok "GAIA questions already present ($Q_COUNT test + $DEV_COUNT dev)"
else
    log "Downloading GAIA questions (needs HF_TOKEN with dataset access)..."
    log "  If this fails, accept the terms at: https://huggingface.co/datasets/gaia-benchmark/GAIA"
    python3 gaia_tools/prep_gaia_data.py && ok "GAIA data ready" \
        || warn "GAIA download failed. Run 'python3 gaia_tools/prep_gaia_data.py' manually later."
fi

# ============================================================================
# Step 8: Verify
# ============================================================================
chmod +x ask 2>/dev/null || true

log "Step 8/8: Verifying installation..."

ERRORS=0

python3 -c "import nat" 2>/dev/null && ok "Python: nat" || { err "Python: nat NOT importable"; ERRORS=$((ERRORS+1)); }
python3 -c "import requests" 2>/dev/null && ok "Python: requests" || { err "Python: requests NOT importable"; ERRORS=$((ERRORS+1)); }
python3 -c "import yaml" 2>/dev/null && ok "Python: yaml" || { err "Python: yaml NOT importable"; ERRORS=$((ERRORS+1)); }

if [ "$MODE" = gpu ]; then
    python3 -c "import vllm" 2>/dev/null && ok "Python: vllm" || { err "Python: vllm NOT importable"; ERRORS=$((ERRORS+1)); }
    nvidia-smi &>/dev/null && ok "NVIDIA GPUs detected" || warn "nvidia-smi failed (vLLM will not work without GPUs)"
else
    command -v ollama &>/dev/null && ok "Ollama CLI available" || { err "Ollama CLI not found"; ERRORS=$((ERRORS+1)); }
    curl -sf http://localhost:11434 &>/dev/null && ok "Ollama server running" || { err "Ollama server not responding on port 11434"; ERRORS=$((ERRORS+1)); }
    ollama list 2>/dev/null | grep -q "^${OLLAMA_MODEL}" && ok "Model ${OLLAMA_MODEL} available" || { err "Model ${OLLAMA_MODEL} not found in ollama list"; ERRORS=$((ERRORS+1)); }
fi

if [ -f ".env" ]; then
    # Verify .env has the required keys (not just that the file exists)
    ENV_KEYS=$(grep -cE '^(TAVILY_API_KEY|NGC_API_KEY|HF_TOKEN)=' .env 2>/dev/null || echo 0)
    if [ "$ENV_KEYS" -ge 2 ]; then
        ok ".env file valid ($ENV_KEYS keys)"
    else
        err ".env file exists but only has $ENV_KEYS keys. Re-run setup to fix."
        ERRORS=$((ERRORS+1))
    fi
else
    err ".env file missing. API keys were not saved."
    ERRORS=$((ERRORS+1))
fi

[ -f "gaia_questions.json" ] && ok "GAIA questions file present" || warn "gaia_questions.json not found (run prep_gaia_data.py)"
[ -d "gaia_files" ] && ok "GAIA files directory present" || warn "gaia_files/ not found (run prep_gaia_data.py)"

command -v stockfish &>/dev/null && ok "Stockfish available" || warn "Stockfish not available"

echo ""
echo "============================================================"
if [ "$ERRORS" -gt 0 ]; then
    err "Setup completed with $ERRORS error(s). Check messages above."
else
    ok "Setup complete! All checks passed."
fi
echo "============================================================"
echo ""

if [ "$MODE" = gpu ]; then
    echo "  Next steps (run from this directory):"
    echo ""
    echo "    1. bash gaia_tools/start_services.sh    # start vLLM + Phoenix"
    echo "    2. ./ask                                 # start chatting with your agent"
else
    echo "  Next steps (run from this directory):"
    echo ""
    echo "    1. ./ask                                 # start chatting"
    echo "       switch ollama                         # load the Ollama agent"
    echo ""
    echo "  The Ollama agent runs locally, no GPU or vLLM needed."
    echo "  Model: ${OLLAMA_MODEL} | Endpoint: http://localhost:11434"
    if [ -z "${NGC_API_KEY:-}" ]; then
        echo ""
        echo "  Note: NGC_API_KEY was skipped. describe_image and transcribe_audio"
        echo "  tools will not work. Add NGC_API_KEY to .env to enable them."
    fi
fi
echo ""
echo "============================================================"
