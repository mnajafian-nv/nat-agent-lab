#!/usr/bin/env bash
# ============================================================================
# gaia_run_all.sh - Run all 3 LOCAL agent architectures sequentially, submit each to HF
#
# This runs single + multi + ultrafast on local vLLM. It does NOT include the
# ultrafast-nogpu agent or custom configs. Use "benchmark ultrafast-nogpu" or
# "benchmark custom" for those.
#
# Usage (in tmux, output shows on screen AND is saved to log):
#   bash gaia_tools/gaia_run_all.sh
#   ORG_NAME=UCB TEAM_NAME=Instructor bash gaia_tools/gaia_run_all.sh
#
# What it does (set ORG_NAME / TEAM_NAME env vars before running):
#   1. Single-agent  -> submits as NAT-<ORG_NAME>-<TEAM_NAME>-SingleAgent
#   2. Multi-agent   -> submits as NAT-<ORG_NAME>-<TEAM_NAME>-MultiAgent
#   3. Ultrafast     -> submits as NAT-<ORG_NAME>-<TEAM_NAME>-Ultrafast
#
# Each run restarts NAT with the appropriate config. vLLM stays running.
# Results are logged to this script's output and to each run's own directory.
# If a run fails, the script logs the error and moves on to the next.
# ============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || { echo "Cannot cd to $REPO_ROOT"; exit 1; }
[[ -f ".venv/bin/activate" ]] && source .venv/bin/activate
if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
fi

# ---- Prompt for org and team name if not set ----
if [[ -z "${ORG_NAME:-}" ]]; then
    read -rp "  Enter your org name (e.g. UCB, Stanford, NVIDIA): " ORG_NAME
    if [[ -z "$ORG_NAME" ]]; then
        echo "  No org name entered. Exiting."
        exit 1
    fi
fi
export ORG_NAME

if [[ -z "${TEAM_NAME:-}" || "${TEAM_NAME:-}" == "YourTeam" ]]; then
    echo ""
    echo "  Leaderboard names will be:"
    echo "    NAT-${ORG_NAME}-<TEAM_NAME>-SingleAgent"
    echo "    NAT-${ORG_NAME}-<TEAM_NAME>-MultiAgent"
    echo "    NAT-${ORG_NAME}-<TEAM_NAME>-Ultrafast"
    echo ""
    read -rp "  Enter your team name (e.g. Instructor, AgentSmiths): " TEAM_NAME
    if [[ -z "$TEAM_NAME" ]]; then
        echo "  No name entered. Exiting."
        exit 1
    fi
fi
export TEAM_NAME

# ---- Capture remaining args (e.g., --level 2) to forward to each run ----
EXTRA_ARGS=("$@")

# ---------------------------------------------------------------------------
# Auto-logging: write to screen AND log file, with line buffering so output
# appears immediately. No external "| tee" pipe needed.
# ---------------------------------------------------------------------------
LOG_FILE="${REPO_ROOT}/logs/gaia_run_all_$(date +%Y%m%d_%H%M).log"
mkdir -p "${REPO_ROOT}/logs"
if command -v stdbuf &>/dev/null; then
    exec > >(stdbuf -oL tee "$LOG_FILE") 2>&1
else
    exec > >(tee "$LOG_FILE") 2>&1
fi
echo "Logging to: $LOG_FILE"

TIMEOUT=360
TAG_PREFIX="run_$(date +%Y%m%d_%H%M)"

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }

RESULTS_SUMMARY=""

run_agent() {
    local mode="$1"
    local tag="${TAG_PREFIX}_${mode}"

    log "============================================================"
    log "Starting $mode run (tag: $tag)"
    log "============================================================"

    local start_time=$SECONDS

    if bash gaia_tools/gaia_run.sh --${mode} --tag "$tag" --timeout "$TIMEOUT" "${EXTRA_ARGS[@]}"; then
        local elapsed=$(( SECONDS - start_time ))
        ok "$mode completed in ${elapsed}s"

        local run_dir
        case "$mode" in
            single)    run_dir="single-agent/runs/latest" ;;
            multi)     run_dir="multi-agent/runs/latest" ;;
            ultrafast) run_dir="ultrafast-agent/runs/latest" ;;
        esac

        local score_info
        score_info=$(python3 -c "
import json
try:
    d=json.load(open('$run_dir/gaia_summary.json'))
    if 'hf_score' in d:
        print(f'{d[\"hf_correct\"]}/{d[\"hf_total\"]} correct ({d[\"hf_score\"]}%)')
    elif 'raw_correct' in d:
        print(f'{d[\"raw_correct\"]}/{d[\"raw_total\"]} correct')
    else:
        print(f'{d.get(\"answered\",\"?\")}/{d.get(\"total\",\"?\")} attempted')
except: print('see logs')
" 2>/dev/null)

        RESULTS_SUMMARY="${RESULTS_SUMMARY}\n  $mode: ${score_info:-see logs} (${elapsed}s)"
    else
        local elapsed=$(( SECONDS - start_time ))
        err "$mode FAILED after ${elapsed}s"
        RESULTS_SUMMARY="${RESULTS_SUMMARY}\n  $mode: FAILED (${elapsed}s)"
    fi

    log ""
}

log "============================================================"
log "  GAIA Full Benchmark - 3 Local Agents (single + multi + ultrafast)"
log "  Started:  $(date)"
log "  Team:     ${TEAM_NAME:-YourTeam}"
log "  Timeout:  ${TIMEOUT}s per question"
log "  Tag:      ${TAG_PREFIX}"
log "============================================================"
log ""

run_agent "single"
run_agent "multi"
run_agent "ultrafast"

log ""
log "============================================================"
log "  ALL RUNS COMPLETE - $(date)"
log "============================================================"
log "Results summary:${RESULTS_SUMMARY}"
log ""
log "Check detailed results:"
log "  single:    single-agent/runs/latest/gaia_summary.json"
log "  multi:     multi-agent/runs/latest/gaia_summary.json"
log "  ultrafast: ultrafast-agent/runs/latest/gaia_summary.json"
log "============================================================"
