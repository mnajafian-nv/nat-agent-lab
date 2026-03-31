#!/usr/bin/env bash
# ============================================================================
# start_services.sh - Start all services needed for GAIA benchmarks
#
# Usage (run from any tmux session on Brev):
#   bash gaia_tools/start_services.sh            # start everything
#   bash gaia_tools/start_services.sh --check    # just check status
#   bash gaia_tools/start_services.sh --stop     # stop everything
#   bash gaia_tools/start_services.sh --phoenix  # start only Phoenix
#
# Creates tmux sessions:
#   vllm    - vLLM model server (port 9000), stays up between runs
#   phoenix - Phoenix tracing UI (port 6006), optional
#   gaia    - your interactive session for running benchmarks
#
# After running this, attach to the gaia session:
#   tmux attach -t gaia
# ============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi
VLLM_PORT=9000
NAT_PORT=8000
PHOENIX_PORT=6006

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*"; }
log()  { echo -e "${BLUE}[INFO]${NC} $*"; }

check_vllm()    { curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; }
check_nat()     { curl -sf "http://localhost:${NAT_PORT}/health" > /dev/null 2>&1; }
check_phoenix() { curl -sf "http://localhost:${PHOENIX_PORT}" > /dev/null 2>&1; }

print_status() {
    echo ""
    log "Service status:"
    check_vllm    && ok "  vLLM     :${VLLM_PORT}" || warn "  vLLM     :${VLLM_PORT} DOWN"
    check_nat     && ok "  NAT      :${NAT_PORT}"  || warn "  NAT      :${NAT_PORT} DOWN (started by gaia_run.sh)"
    check_phoenix && ok "  Phoenix  :${PHOENIX_PORT}" || warn "  Phoenix  :${PHOENIX_PORT} DOWN"
    echo ""
    log "tmux sessions:"
    tmux ls 2>/dev/null || echo "  (none)"
    echo ""
}

# ---- Stop mode ----
if [[ "${1:-}" == "--stop" ]]; then
    log "Stopping all services..."
    pkill -f "nat serve" 2>/dev/null && ok "NAT stopped" || warn "NAT was not running"
    pkill -f "phoenix" 2>/dev/null && ok "Phoenix stopped" || warn "Phoenix was not running"
    tmux kill-session -t phoenix 2>/dev/null && ok "Phoenix tmux killed" || warn "Phoenix tmux was not running"
    pkill -9 -f "vllm serve" 2>/dev/null && ok "vLLM killed" || warn "vLLM was not running"
    tmux kill-session -t vllm 2>/dev/null && ok "vLLM tmux killed" || warn "vLLM tmux was not running"
    rm -rf /dev/shm/* 2>/dev/null && ok "Shared memory cleaned" || true
    if command -v fuser &>/dev/null; then
        for dev in /dev/nvidia*; do fuser -k "$dev" 2>/dev/null || true; done
    fi
    nvidia-smi --gpu-reset 2>/dev/null && ok "GPU state reset" || warn "GPU reset not supported (ok)"
    echo ""
    log "Services stopped. The 'gaia' tmux session is still alive."
    exit 0
fi

# ---- Phoenix-only mode ----
if [[ "${1:-}" == "--phoenix" ]]; then
    cd "$REPO_ROOT" || { err "Cannot cd to $REPO_ROOT"; exit 1; }
    if check_phoenix; then
        ok "Phoenix already running on port $PHOENIX_PORT"
    else
        mkdir -p "$REPO_ROOT/logs"
        pkill -f "phoenix.server.main" 2>/dev/null || true
        tmux kill-session -t phoenix 2>/dev/null || true
        sleep 1

        cat > /tmp/start_phoenix.sh << PHOENIX_EOF
#!/usr/bin/env bash
cd $REPO_ROOT
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
echo "Starting Phoenix on port ${PHOENIX_PORT}..."
PHOENIX_PORT=${PHOENIX_PORT} python3 -m phoenix.server.main serve 2>&1 | tee $REPO_ROOT/logs/phoenix.log
echo "Phoenix exited. Check $REPO_ROOT/logs/phoenix.log"
sleep 9999
PHOENIX_EOF
        chmod +x /tmp/start_phoenix.sh
        tmux new-session -d -s phoenix "bash /tmp/start_phoenix.sh"

        log "Phoenix starting in tmux session 'phoenix'. Waiting..."
        WAITED=0
        while ! check_phoenix; do
            sleep 2
            WAITED=$((WAITED + 2))
            if [ $WAITED -ge 30 ]; then
                warn "Phoenix not healthy after 30s. Check: tmux attach -t phoenix"
                exit 1
            fi
            printf "."
        done
        echo ""
        ok "Phoenix healthy after ${WAITED}s (tmux session: phoenix)"
    fi
    exit 0
fi

# ---- Check mode ----
if [[ "${1:-}" == "--check" ]]; then
    print_status

    # Check system deps
    command -v stockfish &>/dev/null && ok "  Stockfish installed" || warn "  Stockfish NOT installed (run: sudo apt-get install -y stockfish)"

    # Check API keys
    if [[ -z "${TAVILY_API_KEY:-}" ]]; then
        err "TAVILY_API_KEY not set. Export it: export TAVILY_API_KEY='tvly-...'"
    else
        ok "  TAVILY_API_KEY set (${TAVILY_API_KEY:0:8}...)"
    fi
    if [[ -z "${NGC_API_KEY:-}" ]]; then
        err "NGC_API_KEY not set. Export it: export NGC_API_KEY='nvapi-...'"
    else
        ok "  NGC_API_KEY set (${NGC_API_KEY:0:8}...)"
    fi
    if [[ -z "${HF_TOKEN:-}" ]]; then
        warn "HF_TOKEN not set. Leaderboard submission may fail."
        echo "    Get a token at: https://huggingface.co/settings/tokens"
    else
        ok "  HF_TOKEN set (${HF_TOKEN:0:8}...)"
    fi
    exit 0
fi

# ============================================================================
# Start services
# ============================================================================

cd "$REPO_ROOT" || { err "Cannot cd to $REPO_ROOT"; exit 1; }

# 1. Install system dependencies
log "Checking system dependencies..."
if command -v stockfish &>/dev/null; then
    ok "Stockfish already installed ($(stockfish --help 2>&1 | head -1 || echo 'available'))"
elif command -v apt-get &>/dev/null; then
    log "Installing Stockfish chess engine (needed by solve_chess tool)..."
    sudo apt-get install -y -qq stockfish 2>/dev/null \
        && ok "Stockfish installed" \
        || warn "Stockfish install failed (solve_chess will use checkmate search only)"
else
    warn "Stockfish not available (apt-get not found; macOS users: brew install stockfish)"
fi

# 2. Check API keys
log "Checking API keys..."
if [[ -z "${TAVILY_API_KEY:-}" || -z "${NGC_API_KEY:-}" ]]; then
    err "Required API keys not set. Add them to $REPO_ROOT/.env:"
    echo "  TAVILY_API_KEY='tvly-...'"
    echo "  NGC_API_KEY='nvapi-...'"
    echo "  HF_TOKEN='hf_...'"
    exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    warn "HF_TOKEN not set. Leaderboard submission may fail."
    warn "  Get a token at: https://huggingface.co/settings/tokens"
fi
ok "API keys set"

# 3. Start vLLM (in its own tmux session)
log "Starting vLLM..."
if check_vllm; then
    ok "vLLM already running on port $VLLM_PORT"
else
    # Clean up stale resources from previous crashed vLLM
    log "Cleaning up stale vLLM processes, shared memory, and GPU state..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
    tmux kill-session -t vllm 2>/dev/null || true
    rm -rf /dev/shm/* 2>/dev/null || true
    # Kill any zombie processes holding GPU memory
    if command -v fuser &>/dev/null; then
        for dev in /dev/nvidia*; do
            fuser -k "$dev" 2>/dev/null || true
        done
    fi
    nvidia-smi --gpu-reset 2>/dev/null || true
    sleep 3

    cat > /tmp/start_vllm.sh << VLLM_EOF
#!/usr/bin/env bash
cd $REPO_ROOT
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TAVILY_API_KEY='$TAVILY_API_KEY'
export NGC_API_KEY='$NGC_API_KEY'
NCCL_DEBUG=WARN SAFETENSORS_FAST_GPU=1 vllm serve \\
    MiniMaxAI/MiniMax-M2.5 --trust-remote-code \\
    --enable_expert_parallel --tensor-parallel-size 8 \\
    --enable-auto-tool-choice --tool-call-parser minimax_m2 \\
    --reasoning-parser minimax_m2_append_think \\
    --port $VLLM_PORT
VLLM_EOF
    chmod +x /tmp/start_vllm.sh

    mkdir -p "$REPO_ROOT/logs"
    tmux new-session -d -s vllm "bash /tmp/start_vllm.sh 2>&1 | tee $REPO_ROOT/logs/vllm.log; echo 'vLLM exited. Check $REPO_ROOT/logs/vllm.log'; sleep 9999"

    log "vLLM starting in tmux session 'vllm'. Waiting for health..."
    WAITED=0
    while ! check_vllm; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $WAITED -ge 300 ]; then
            err "vLLM not healthy after 300s. Check: tmux attach -t vllm"
            exit 1
        fi
        printf "."
    done
    echo ""
    ok "vLLM healthy after ${WAITED}s"
fi

# 4. Start Phoenix (in its own tmux session)
log "Starting Phoenix..."
if check_phoenix; then
    ok "Phoenix already running on port $PHOENIX_PORT"
else
    mkdir -p "$REPO_ROOT/logs"
    pkill -f "phoenix.server.main" 2>/dev/null || true
    tmux kill-session -t phoenix 2>/dev/null || true
    sleep 1

    cat > /tmp/start_phoenix.sh << PHOENIX_EOF
#!/usr/bin/env bash
cd $REPO_ROOT
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
echo "Starting Phoenix on port ${PHOENIX_PORT}..."
PHOENIX_PORT=${PHOENIX_PORT} python3 -m phoenix.server.main serve 2>&1 | tee $REPO_ROOT/logs/phoenix.log
echo "Phoenix exited. Check $REPO_ROOT/logs/phoenix.log"
sleep 9999
PHOENIX_EOF
    chmod +x /tmp/start_phoenix.sh

    tmux new-session -d -s phoenix "bash /tmp/start_phoenix.sh"

    log "Phoenix starting in tmux session 'phoenix'. Waiting for health..."
    WAITED=0
    while ! check_phoenix; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $WAITED -ge 30 ]; then
            warn "Phoenix not healthy after 30s (non-critical, benchmark still works)"
            warn "  Check: tmux attach -t phoenix"
            break
        fi
        printf "."
    done
    if check_phoenix; then
        echo ""
        ok "Phoenix healthy after ${WAITED}s"
    fi
fi

# 5. Final status
print_status
log "Ready! Start exploring:"
echo ""
echo "  ./ask                                    # interactive REPL (recommended)"
echo ""
echo "  Or run benchmarks directly:"
echo "  bash gaia_tools/gaia_run.sh --single    # single-agent"
echo "  bash gaia_tools/gaia_run.sh --multi     # multi-agent"
echo "  bash gaia_tools/gaia_run.sh --ultrafast # ultrafast-agent"
echo "  bash gaia_tools/gaia_run_all.sh         # all 3 sequentially"
echo ""
