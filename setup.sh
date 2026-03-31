#!/usr/bin/env bash
# ============================================================================
# setup.sh - One-time environment setup for GAIA Agent Toolkit
#
# Installs all dependencies, downloads the model, and verifies the environment.
# Safe to re-run (all steps are idempotent).
#
# Prerequisites (local LLM):
#   - Linux with 8x H100 GPUs (or equivalent)
#   - 300 GB free disk space in the cloned directory
#   - NVIDIA driver installed (nvidia-smi works)
#
# Prerequisites (cloud LLM - use --cloud flag):
#   - Any Linux machine (no GPU required for the LLM)
#   - NGC_API_KEY with access to build.nvidia.com
#
# Usage:
#   cd <repo-root>
#   bash setup.sh            # full setup (local vLLM + model download)
#   bash setup.sh --cloud    # cloud-only (skips vLLM, model download, disk check)
# ============================================================================
set -uo pipefail

# ---- Auto-wrap in tmux to survive SSH disconnects ----
if [ -z "${TMUX:-}" ] && command -v tmux &>/dev/null; then
    tmux kill-session -t setup 2>/dev/null || true
    echo "  Starting setup in tmux session 'setup' (survives SSH disconnects)..."
    exec tmux new-session -s setup "bash \"$0\" $*; echo; echo 'Setup complete. Press Enter to close.'; read"
fi

CLOUD_MODE=false
for arg in "$@"; do
    if [[ "$arg" == "--cloud" ]]; then
        CLOUD_MODE=true
    fi
done

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }
die()  { err "$*"; exit 1; }

echo ""
echo "============================================================"
if $CLOUD_MODE; then
    echo "  NAT Agent Lab - Cloud Setup (no local GPU needed)"
else
    echo "  NAT Agent Lab - Environment Setup"
fi
echo "============================================================"
echo "  Repo: $REPO_ROOT"
echo ""

# ============================================================================
# Pre-flight: Auto-detect missing GPU and suggest --cloud
# ============================================================================
if ! $CLOUD_MODE && ! nvidia-smi &>/dev/null; then
    warn "No NVIDIA GPU detected (nvidia-smi not found)."
    echo "  You cannot run the local vLLM model without GPUs."
    echo ""
    echo "  Re-run with:  bash setup.sh --cloud"
    echo ""
    echo "  The --cloud flag uses NVIDIA Build instead of local vLLM."
    echo "  Same tools, same architecture, no GPU needed (uses Qwen 3.5-122B on NVIDIA Build)."
    echo ""
    die "Aborting. Re-run with: bash setup.sh --cloud"
fi

# ============================================================================
# Step 1: Disk space check (skipped in --cloud mode)
# ============================================================================
if $CLOUD_MODE; then
    log "Step 1/8: Disk space check skipped (cloud mode)"
else
    log "Step 1/8: Checking disk space..."
    AVAIL_GB=$(df -BG "$REPO_ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
    if [ "$AVAIL_GB" -lt 300 ]; then
        die "Only ${AVAIL_GB}GB free. Need at least 300GB for model weights.
  Clone the repo to a directory with more space (e.g., /ephemeral/, /data/).
  Or run: bash setup.sh --cloud  (skips local model, uses NVIDIA Build instead)"
    fi
    ok "Disk space: ${AVAIL_GB}GB available"
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

source .venv/bin/activate
ok "Python $(python3 --version) in .venv"

# ============================================================================
# Step 3: Install Python packages
# ============================================================================
log "Step 3/8: Installing Python packages..."

if python3 -c "import nat; import datasets; import openpyxl; import bs4; import pypdf; import pptx; import sympy" 2>/dev/null; then
    ok "NAT and dependencies already installed"
else
    log "Installing NAT and dependencies..."
    uv pip install "nvidia-nat[langchain,phoenix]" arize-phoenix requests pyyaml datasets \
        openpyxl beautifulsoup4 pypdf python-pptx sympy 2>/dev/null \
        || pip install "nvidia-nat[langchain,phoenix]" arize-phoenix requests pyyaml datasets \
        openpyxl beautifulsoup4 pypdf python-pptx sympy
    ok "NAT installed"
fi

if python3 -c "from gaia_tools.register import read_file" 2>/dev/null; then
    ok "GAIA custom tools already installed"
elif [ -f "gaia_tools/pyproject.toml" ]; then
    log "Installing GAIA custom tools (read_file, fetch_url, python_executor, etc.)..."
    uv pip install -e gaia_tools/ 2>/dev/null || pip install -e gaia_tools/
    ok "GAIA tools installed"
fi

if $CLOUD_MODE; then
    log "  vLLM install skipped (cloud mode)"
else
    if python3 -c "import vllm" 2>/dev/null; then
        ok "vLLM already installed"
    else
        log "Installing vLLM (this may take a few minutes)..."
        uv pip install vllm --torch-backend=auto 2>/dev/null \
            || pip install vllm
        ok "vLLM installed"
    fi
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
else
    warn "Stockfish not available (apt-get not found; macOS users: brew install stockfish)"
fi

# ============================================================================
# Step 5: API keys
# ============================================================================
log "Step 5/8: Checking API keys..."

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    ok ".env file found and loaded"
fi

NEEDS_KEYS=false

if [ -z "${TAVILY_API_KEY:-}" ]; then
    echo ""
    echo "  Tavily API key is needed for internet search."
    echo "  Get one free at: https://tavily.com/"
    read -rp "  TAVILY_API_KEY: " TAVILY_API_KEY
    [ -z "$TAVILY_API_KEY" ] && die "Tavily key is required."
    NEEDS_KEYS=true
else
    ok "TAVILY_API_KEY set (${TAVILY_API_KEY:0:8}...)"
fi

if [ -z "${NGC_API_KEY:-}" ]; then
    echo ""
    echo "  NVIDIA Build API key is needed for vision models."
    echo "  Get one at: https://build.nvidia.com/"
    read -rp "  NGC_API_KEY: " NGC_API_KEY
    [ -z "$NGC_API_KEY" ] && die "NGC key is required."
    NEEDS_KEYS=true
else
    ok "NGC_API_KEY set (${NGC_API_KEY:0:8}...)"
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo ""
    echo "  HuggingFace token is needed for GAIA dataset and leaderboard submission."
    echo "  Get one at: https://huggingface.co/settings/tokens"
    read -rp "  HF_TOKEN: " HF_TOKEN
    [ -z "$HF_TOKEN" ] && die "HuggingFace token is required."
    NEEDS_KEYS=true
else
    ok "HF_TOKEN set (${HF_TOKEN:0:8}...)"
fi

if $NEEDS_KEYS; then
    # Write each key individually (preserves other entries like HF_HOME).
    # Uses Python for sed-free portability (macOS sed -i is incompatible with GNU).
    python3 -c "
import os, re
keys = {
    'TAVILY_API_KEY': '''${TAVILY_API_KEY}''',
    'NGC_API_KEY': '''${NGC_API_KEY}''',
    'HF_TOKEN': '''${HF_TOKEN}''',
}
env_path = '.env'
lines = open(env_path).readlines() if os.path.exists(env_path) else []
for key, val in keys.items():
    pattern = re.compile(r'^(export\s+)?' + re.escape(key) + r'=.*$')
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
    ok "Keys saved to .env"
fi

export TAVILY_API_KEY NGC_API_KEY HF_TOKEN

# ============================================================================
# Step 6: Download model weights (skipped in --cloud mode)
# ============================================================================
if $CLOUD_MODE; then
    log "Step 6/8: Model download skipped (cloud mode, using NVIDIA Build)"
else
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
fi

# ============================================================================
# Step 7: GAIA questions and files
# ============================================================================
# Ensure HF cache is local and writable (cloud .env may have a stale HF_HOME
# from another machine, e.g. /ephemeral on a GPU instance).
if [ -z "${HF_HOME:-}" ] || ! mkdir -p "$HF_HOME" 2>/dev/null; then
    export HF_HOME="$REPO_ROOT/.cache/huggingface"
    mkdir -p "$HF_HOME"
fi

log "Step 7/8: Checking GAIA questions..."

if [ -f "gaia_questions.json" ]; then
    Q_COUNT=$(python3 -c "import json; print(len(json.load(open('gaia_questions.json'))))" 2>/dev/null || echo "0")
    ok "GAIA questions already present ($Q_COUNT questions)"
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
if ! $CLOUD_MODE; then
    python3 -c "import vllm" 2>/dev/null && ok "Python: vllm" || { err "Python: vllm NOT importable"; ERRORS=$((ERRORS+1)); }
fi
python3 -c "import requests" 2>/dev/null && ok "Python: requests" || { err "Python: requests NOT importable"; ERRORS=$((ERRORS+1)); }
python3 -c "import yaml" 2>/dev/null && ok "Python: yaml" || { err "Python: yaml NOT importable"; ERRORS=$((ERRORS+1)); }

[ -f "gaia_questions.json" ] && ok "GAIA questions file present" || warn "gaia_questions.json not found (run prep_gaia_data.py)"
[ -d "gaia_files" ] && ok "GAIA files directory present" || warn "gaia_files/ not found (run prep_gaia_data.py)"

command -v stockfish &>/dev/null && ok "Stockfish available" || warn "Stockfish not available"
if $CLOUD_MODE; then
    nvidia-smi &>/dev/null && ok "NVIDIA GPUs detected (optional for cloud)" || ok "No local GPU (using cloud LLM)"
else
    nvidia-smi &>/dev/null && ok "NVIDIA GPUs detected" || err "nvidia-smi failed"
fi

echo ""
echo "============================================================"
if [ "$ERRORS" -gt 0 ]; then
    err "Setup completed with $ERRORS error(s). Check messages above."
else
    ok "Setup complete! All checks passed."
fi
echo "============================================================"
echo ""
echo "  Next steps (run from this directory):"
echo ""
if $CLOUD_MODE; then
    echo "    1. ./ask                                 # launch the REPL (auto-selects ultrafast-nogpu)"
else
    echo "    1. bash gaia_tools/start_services.sh    # start vLLM + Phoenix"
    echo "    2. ./ask                                 # launch the interactive REPL"
fi
echo ""
echo "============================================================"
