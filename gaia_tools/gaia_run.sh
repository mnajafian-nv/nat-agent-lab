#!/usr/bin/env bash
# ============================================================================
# gaia_run.sh - One-command GAIA benchmark runner with error recovery
#
# Usage:
#   ./gaia_run.sh --single                         # single-agent, auto-submit
#   ./gaia_run.sh --multi                           # multi-agent (orchestrator), auto-submit
#   ./gaia_run.sh --ultrafast                       # ultrafast single-agent w/ routing prompt
#   ./gaia_run.sh --ultrafast-nogpu                  # ultrafast-nogpu (NVIDIA Build, no local GPU)
#   ./gaia_run.sh -c path/to/config.yml             # custom config
#   ./gaia_run.sh --single --tag "after chess fix"  # label the run
#   ./gaia_run.sh --single -u MyTeam --limit 5      # quick 5-question test
#   ./gaia_run.sh --single --serve                  # start NAT only (no benchmark)
#   ./gaia_run.sh --dry-run --single                # validate config only
#   ./gaia_run.sh --status                          # check all services
#   ./gaia_run.sh --history                         # show all past runs
#
# Each run is auto-numbered (run1, run2, run3...) per agent folder:
#   single-agent/runs/run14_gaia_agent/
#   multi-agent/runs/run3_gaia_agent_multi/
#   ultrafast-agent/runs/run1_gaia_agent_ultrafast/
#
# Leaderboard usernames (set ORG_NAME / TEAM_NAME env vars to customize):
#   --single     ->  NAT-<ORG_NAME>-<TEAM_NAME>-SingleAgent
#   --multi      ->  NAT-<ORG_NAME>-<TEAM_NAME>-MultiAgent
#   --ultrafast  ->  NAT-<ORG_NAME>-<TEAM_NAME>-Ultrafast
#   --ultrafast-nogpu -> NAT-<ORG_NAME>-<TEAM_NAME>-Ultrafast-NoGPU
# ============================================================================
set -uo pipefail

# ---- Resolve repo root from script location ----
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Defaults ----
ORG_NAME="${ORG_NAME:-}"
TEAM_NAME="${TEAM_NAME:-}"
CONFIG=""
USERNAME=""
SUBMIT="--submit"
NAT_PORT=8000
VLLM_PORT=9000
NAT_TIMEOUT=45
VLLM_TIMEOUT=60
BENCHMARK_TIMEOUT=300
EXTRA_ARGS=""
DRY_RUN=false
STATUS_ONLY=false
HISTORY_ONLY=false
SERVE_ONLY=false
RUN_TAG=""

# ---- Colors ----
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }
die()  { err "$*"; exit 1; }

# ---- Parse arguments ----
AGENT_MODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --single)
            CONFIG="single-agent/gaia_agent.yml"
            AGENT_MODE="SingleAgent"
            shift ;;
        --multi)
            CONFIG="multi-agent/gaia_agent_multi.yml"
            AGENT_MODE="MultiAgent"
            shift ;;
        --ultrafast)
            CONFIG="ultrafast-agent/gaia_agent_ultrafast.yml"
            AGENT_MODE="Ultrafast"
            shift ;;
        --ultrafast-nogpu)
            CONFIG="ultrafast-nogpu-agent/gaia_agent_ultrafast_nogpu.yml"
            AGENT_MODE="Ultrafast-NoGPU"
            shift ;;
        -c|--config)    CONFIG="$2"; shift 2 ;;
        -u|--username)  USERNAME="$2"; shift 2 ;;
        --submit)       SUBMIT="--submit"; shift ;;
        --no-submit)    SUBMIT="--no-submit"; shift ;;
        --limit)        EXTRA_ARGS="$EXTRA_ARGS --limit $2"; shift 2 ;;
        --level)        EXTRA_ARGS="$EXTRA_ARGS --level $2"; shift 2 ;;
        --timeout)      BENCHMARK_TIMEOUT="$2"; shift 2 ;;
        --tag)          RUN_TAG="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --serve)        SERVE_ONLY=true; shift ;;
        --status)       STATUS_ONLY=true; shift ;;
        --history)      HISTORY_ONLY=true; shift ;;
        -h|--help)
            head -20 "$0" | tail -18
            exit 0 ;;
        *) die "Unknown argument: $1. Use --help." ;;
    esac
done

# ---- Resolve paths ----
cd "$REPO_ROOT" || die "Cannot cd to $REPO_ROOT"
[[ -f ".venv/bin/activate" ]] && source .venv/bin/activate
if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
fi
# Some NAT components (vision tools, etc.) look for NVIDIA_API_KEY
[[ -n "${NGC_API_KEY:-}" && -z "${NVIDIA_API_KEY:-}" ]] && export NVIDIA_API_KEY="$NGC_API_KEY"

# ============================================================================
# Health check functions
# ============================================================================

check_vllm() {
    curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1
}

check_nat() {
    curl -sf "http://localhost:${NAT_PORT}/health" > /dev/null 2>&1
}

check_phoenix() {
    curl -sf "http://localhost:6006" > /dev/null 2>&1
}

print_status() {
    echo ""
    log "Service status:"
    check_vllm    && ok "  vLLM     :${VLLM_PORT}" || warn "  vLLM     :${VLLM_PORT} DOWN"
    check_nat     && ok "  NAT      :${NAT_PORT}"  || warn "  NAT      :${NAT_PORT} DOWN"
    check_phoenix && ok "  Phoenix  :6006"          || warn "  Phoenix  :6006 DOWN"
    echo ""
}

if $STATUS_ONLY; then
    print_status
    exit 0
fi

# ---- History: show all past runs ----
if $HISTORY_ONLY; then
    echo ""
    log "Run history:"
    echo ""
    # Show history for all experiment folders that have runs
    for exp_dir in single-agent multi-agent ultrafast-agent ultrafast-nogpu-agent; do
        [[ -d "$exp_dir/runs" ]] || continue
        summaries=$(ls -d "$exp_dir"/runs/run*/ 2>/dev/null | sort -V)
        [[ -z "$summaries" ]] && continue
        echo "  $exp_dir:"
        for run_dir in $summaries; do
            run_name=$(basename "$run_dir")
            summary="$run_dir/gaia_summary.json"
            tag_file="$run_dir/tag.txt"
            tag=""
            [[ -f "$tag_file" ]] && tag=" [$(cat "$tag_file")]"
            if [[ -f "$summary" ]]; then
                score=$(python3 -c "
import json
try:
    d = json.load(open('$summary'))
    ts = d.get('wall_clock_seconds', d.get('total_time_seconds', '?'))
    if 'hf_score' in d:
        print(f'{d[\"hf_correct\"]}/{d[\"hf_total\"]} correct, {d[\"hf_score\"]}% ({ts:.0f}s)')
    elif 'raw_correct' in d:
        print(f'{d[\"raw_correct\"]}/{d[\"raw_total\"]} correct ({ts:.0f}s)')
    else:
        a = d.get('answered', '?')
        t = d.get('total', '?')
        print(f'{a}/{t} attempted, score unknown ({ts:.0f}s)')
except: print('?')
" 2>/dev/null)
                printf "    %-30s %s%s\n" "$run_name" "$score" "$tag"
            else
                if [[ -f "$run_dir/benchmark.log" ]]; then
                    printf "    %-30s (interrupted)%s\n" "$run_name" "$tag"
                else
                    printf "    %-30s (no results)%s\n" "$run_name" "$tag"
                fi
            fi
        done
        echo ""
    done
    exit 0
fi

# ---- Validate config was specified ----
if [[ -z "$CONFIG" ]]; then
    die "No config specified. Use --single, --multi, --ultrafast, --ultrafast-nogpu, or -c <path>.
  Examples:
    bash gaia_tools/gaia_run.sh --single
    bash gaia_tools/gaia_run.sh --multi
    bash gaia_tools/gaia_run.sh --ultrafast
    bash gaia_tools/gaia_run.sh --ultrafast-nogpu
    bash gaia_tools/gaia_run.sh -c single-agent/gaia_agent.yml"
fi

[[ -f "$CONFIG" ]] || die "Config not found: $CONFIG"

# ---- Dry-run: validate config and exit with no side effects ----
if $DRY_RUN; then
    log "Dry run: validating $CONFIG..."
    python3 -c "
import yaml, sys
try:
    data = yaml.safe_load(open('$CONFIG'))
    if not isinstance(data, dict):
        print('ERROR: YAML did not parse to a dict')
        sys.exit(1)
    wf = data.get('workflow', {})
    wf_type = wf.get('_type', '?')
    funcs = list(data.get('functions', {}).keys())
    agents = [k for k,v in data.get('functions', {}).items() if isinstance(v, dict) and '_type' in v and 'agent' in str(v.get('_type',''))]
    print(f'OK: workflow={wf_type}, llm={wf.get(\"llm_name\",\"?\")}, tools={len(funcs)-len(agents)}, agents={len(agents)}')
except yaml.YAMLError as e:
    print(f'YAML PARSE ERROR: {e}')
    sys.exit(1)
" || die "Config validation failed. Fix $CONFIG before running."
    print_status
    ok "Dry run complete. Config is valid: $CONFIG"
    exit 0
fi

# ---- Serve-only: skip team name, run counter, and benchmark ----
if $SERVE_ONLY; then
    log "Serve-only mode: starting NAT with $CONFIG"
else
    # ---- Resolve org, team name, and username ----
    if [[ -z "$USERNAME" ]]; then
        if [[ -z "$ORG_NAME" ]]; then
            read -rp "  Enter your org name (e.g. UCB, Stanford, NVIDIA): " ORG_NAME
            [[ -z "$ORG_NAME" ]] && die "No org name entered."
            export ORG_NAME
        fi
        if [[ -z "$TEAM_NAME" ]]; then
            echo ""
            echo "  Leaderboard name will be: NAT-${ORG_NAME}-<TEAM_NAME>-${AGENT_MODE:-Agent}"
            echo ""
            read -rp "  Enter your team name (e.g. Instructor, AgentSmiths): " TEAM_NAME
            [[ -z "$TEAM_NAME" ]] && die "No team name entered."
            export TEAM_NAME
        fi
        if [[ -n "$AGENT_MODE" ]]; then
            USERNAME="NAT-${ORG_NAME}-${TEAM_NAME}-${AGENT_MODE}"
        else
            USERNAME="NAT-${ORG_NAME}-${TEAM_NAME}"
        fi
    fi
fi

# ---- Derive runs directory from config location ----
CONFIG_DIR=$(dirname "$CONFIG")
CONFIG_BASE="$(basename "$CONFIG" .yml)"

if $SERVE_ONLY; then
    LOG_DIR="/tmp/nat_serve"
    mkdir -p "$LOG_DIR"
    RUN="serve"
else
    # Auto-increment run number per agent folder
    RUNS_DIR="${CONFIG_DIR}/runs"
    mkdir -p "$RUNS_DIR"
    RUN_COUNTER="${RUNS_DIR}/.run_counter"
    if [[ -f "$RUN_COUNTER" ]]; then
        RUN=$(( $(cat "$RUN_COUNTER") + 1 ))
    else
        RUN=1
    fi
    echo "$RUN" > "$RUN_COUNTER"

    LOG_DIR="${RUNS_DIR}/run${RUN}_${CONFIG_BASE}"
    mkdir -p "$LOG_DIR"

    # Save tag if provided
    if [[ -n "$RUN_TAG" ]]; then
        echo "$RUN_TAG" > "$LOG_DIR/tag.txt"
    fi
fi

TAG_DISPLAY=""
[[ -n "$RUN_TAG" ]] && TAG_DISPLAY=" [$RUN_TAG]"

if ! $SERVE_ONLY; then
    log "Run ${RUN} starting (${CONFIG_DIR})${TAG_DISPLAY}"
fi

# ============================================================================
# Step 1: Validate YAML config
# ============================================================================

log "Step 1/5: Validating config: $CONFIG"
python3 -c "
import yaml, sys
try:
    data = yaml.safe_load(open('$CONFIG'))
    if not isinstance(data, dict):
        print('ERROR: YAML did not parse to a dict')
        sys.exit(1)
    wf = data.get('workflow', {})
    wf_type = wf.get('_type', '?')
    sp = wf.get('system_prompt', '')
    if wf_type in ('tool_calling_agent', 'react_agent', 'reasoning_agent', 'rewoo_agent') and len(sp) < 50:
        print(f'WARNING: system_prompt is only {len(sp)} chars (seems too short)')
    for bad in ['\\\\s+', '\\\\d+', '\\\\w+', '\\\\b', '\\\\t']:
        if bad in str(sp):
            print(f'ERROR: system_prompt contains regex escape {bad} which breaks YAML double-quote re-serialization')
            sys.exit(1)
    funcs = list(data.get('functions', {}).keys())
    agents = [k for k,v in data.get('functions', {}).items() if isinstance(v, dict) and '_type' in v and 'agent' in str(v.get('_type',''))]
    print(f'OK: workflow={wf_type}, llm={wf.get(\"llm_name\",\"?\")}, tools={len(funcs)-len(agents)}, agents={len(agents)}')
    if agents:
        print(f'    Sub-agents: {agents}')
except yaml.YAMLError as e:
    print(f'YAML PARSE ERROR: {e}')
    sys.exit(1)
" || die "Config validation failed. Fix $CONFIG before running."

cp "$CONFIG" "$LOG_DIR/config.yml"
# Snapshot all code that affects results (not just the YAML)
cp gaia_tools/gaia_submit.py "$LOG_DIR/gaia_submit.py" 2>/dev/null
cp gaia_tools/src/gaia_tools/register.py "$LOG_DIR/register.py" 2>/dev/null
ok "Config valid: $CONFIG (full snapshot saved to $LOG_DIR/)"

# ============================================================================
# Step 2: Check vLLM (skipped for non-local configs)
# ============================================================================

USES_LOCAL_LLM=$(python3 -c "
import yaml
d = yaml.safe_load(open('$CONFIG'))
wf = d.get('workflow', {})
llm_key = wf.get('llm_name', '')
llm_cfg = d.get('llms', {}).get(llm_key, {})
if llm_cfg.get('_type') == 'nim':
    print('no')
elif 'localhost' not in llm_cfg.get('base_url', 'localhost') and '127.0.0.1' not in llm_cfg.get('base_url', 'localhost'):
    print('no')
else:
    print('yes')
" 2>/dev/null || echo "yes")

export USES_LOCAL_LLM

if [[ "$USES_LOCAL_LLM" == "no" ]]; then
    log "Step 2/5: Skipping vLLM check (non-local model detected)"
    ok "Using NVIDIA Build, no local vLLM needed"
else
    log "Step 2/5: Checking vLLM on port $VLLM_PORT..."
    if check_vllm; then
        ok "vLLM is healthy"
    else
        warn "vLLM not responding. Waiting up to ${VLLM_TIMEOUT}s..."
        for i in $(seq 1 "$VLLM_TIMEOUT"); do
            if check_vllm; then
                ok "vLLM recovered after ${i}s"
                break
            fi
            if [ "$i" -eq "$VLLM_TIMEOUT" ]; then
                die "vLLM not responding after ${VLLM_TIMEOUT}s. Start it manually:
  vllm serve MiniMaxAI/MiniMax-M2.5 --trust-remote-code --tensor-parallel-size 8 \\
    --enable-auto-tool-choice --tool-call-parser minimax_m2 \\
    --reasoning-parser minimax_m2_append_think --port $VLLM_PORT"
            fi
            sleep 1
        done
    fi
fi

# ============================================================================
# Step 3: Restart NAT
# ============================================================================

log "Step 3/5: Restarting NAT with $CONFIG..."

pkill -9 -f "nat serve" 2>/dev/null || true
sleep 3

if lsof -i ":${NAT_PORT}" > /dev/null 2>&1; then
    warn "Port $NAT_PORT still in use. Force-killing..."
    if command -v fuser &>/dev/null; then
        fuser -k "${NAT_PORT}/tcp" 2>/dev/null || true
    else
        lsof -ti ":${NAT_PORT}" | xargs kill -9 2>/dev/null || true
    fi
    sleep 2
fi

NAT_LOG="$LOG_DIR/nat.log"
nohup nat serve --config_file "$CONFIG" --port "$NAT_PORT" > "$NAT_LOG" 2>&1 &
NAT_PID=$!
echo "$NAT_PID" > "$LOG_DIR/nat.pid"
log "NAT started (PID $NAT_PID), waiting for health..."

WAIT=2
TOTAL_WAIT=0
while [ "$TOTAL_WAIT" -lt "$NAT_TIMEOUT" ]; do
    sleep "$WAIT"
    TOTAL_WAIT=$((TOTAL_WAIT + WAIT))

    if ! kill -0 "$NAT_PID" 2>/dev/null; then
        err "NAT process died (PID $NAT_PID). Last 15 lines of log:"
        echo "---"
        tail -15 "$NAT_LOG"
        echo "---"

        if grep -q "unknown escape character" "$NAT_LOG"; then
            die "YAML has regex backslash escapes (e.g. \\s) in system_prompt.
  Remove literal regex from the prompt. Use plain English instead."
        elif grep -q "_dask_client" "$NAT_LOG"; then
            die "NAT _dask_client error. Try: pip install --upgrade dask distributed"
        elif grep -q "Address already in use" "$NAT_LOG"; then
            die "Port $NAT_PORT in use. Run: fuser -k ${NAT_PORT}/tcp"
        else
            die "NAT crashed. Check $NAT_LOG for details."
        fi
    fi

    if check_nat; then
        ok "NAT is healthy after ${TOTAL_WAIT}s (PID $NAT_PID)"
        break
    fi

    log "  ...waiting (${TOTAL_WAIT}s / ${NAT_TIMEOUT}s)"
    WAIT=$((WAIT < 8 ? WAIT * 2 : 8))
done

if ! check_nat; then
    err "NAT not healthy after ${NAT_TIMEOUT}s. Last 15 lines:"
    tail -15 "$NAT_LOG"
    die "NAT failed to start."
fi

# ---- Serve-only: exit after NAT is healthy ----
if $SERVE_ONLY; then
    ok "NAT is running with $CONFIG (PID $NAT_PID, port $NAT_PORT)"
    ok "Log: $NAT_LOG"
    exit 0
fi

# ============================================================================
# Step 4: Run benchmark
# ============================================================================

log "Step 4/5: Running GAIA benchmark (run ${RUN})..."
log "  Run:      ${RUN}${TAG_DISPLAY}"
log "  Username: $USERNAME"
log "  Config:   $CONFIG"
log "  Submit:   $SUBMIT"
log "  Extra:    ${EXTRA_ARGS:-none}"
log "  Logs:     $LOG_DIR/"
echo ""

BENCH_LOG="$LOG_DIR/benchmark.log"
python3 -u gaia_tools/gaia_submit.py "$USERNAME" \
    $SUBMIT \
    --timeout "$BENCHMARK_TIMEOUT" \
    $EXTRA_ARGS \
    2>&1 | if command -v stdbuf &>/dev/null; then stdbuf -oL tee "$BENCH_LOG"; else tee "$BENCH_LOG"; fi

BENCH_EXIT=${PIPESTATUS[0]}

# ============================================================================
# Step 5: Collect results
# ============================================================================

log "Step 5/5: Collecting results..."

for f in gaia_results.json gaia_summary.json; do
    [[ -f "$f" ]] && mv "$f" "$LOG_DIR/"
done

# Inject run metadata into summary
if [[ -f "$LOG_DIR/gaia_summary.json" ]]; then
    python3 -c "
import json, pathlib
with open('$LOG_DIR/gaia_summary.json') as f:
    d = json.load(f)
d['run'] = $RUN
d['config'] = '$CONFIG'
tag_path = pathlib.Path('$LOG_DIR/tag.txt')
d['tag'] = tag_path.read_text().strip() if tag_path.exists() else ''
with open('$LOG_DIR/gaia_summary.json', 'w') as f:
    json.dump(d, f, indent=2)
" 2>/dev/null || true
fi

echo ""
echo "  ────────────────────────────────────────────────────────────"
echo "  Run ${RUN} saved to: $LOG_DIR/"
echo "  Leaderboard:  https://huggingface.co/spaces/agents-course/Students_Leaderboard"
echo "  Run history:  bash gaia_tools/gaia_run.sh --history"
echo "  ────────────────────────────────────────────────────────────"

# Compare with previous run in the SAME experiment folder
PREV_RUN=$(ls -d "${CONFIG_DIR}"/runs/run*_*/ 2>/dev/null | sort -V | grep -v "$(basename "$LOG_DIR")" | tail -1)
if [[ -n "$PREV_RUN" && -f "${PREV_RUN}gaia_summary.json" && -f "$LOG_DIR/gaia_summary.json" ]]; then
    echo ""
    log "Comparing run ${RUN} with previous: $(basename "$PREV_RUN")"
    python3 -c "
import json
def fmt(d):
    if 'hf_score' in d:
        return f'{d[\"hf_correct\"]}/{d[\"hf_total\"]} correct ({d[\"hf_score\"]}%)'
    elif 'raw_correct' in d:
        return f'{d[\"raw_correct\"]}/{d[\"raw_total\"]} correct'
    else:
        return f'{d.get(\"answered\",\"?\")}/{d.get(\"total\",\"?\")} attempted'
def score_val(d):
    return d.get('hf_correct', d.get('raw_correct', None))
try:
    prev = json.load(open('${PREV_RUN}gaia_summary.json'))
    curr = json.load(open('$LOG_DIR/gaia_summary.json'))
    prev_r = prev.get('run', prev.get('version', '?'))
    print(f'  run {prev_r}: {fmt(prev)}')
    print(f'  run ${RUN}: {fmt(curr)}')
    ps, cs = score_val(prev), score_val(curr)
    if isinstance(ps, (int, float)) and isinstance(cs, (int, float)):
        diff = cs - ps
        arrow = '+' if diff > 0 else ''
        print(f'  Delta:    {arrow}{diff}')
except Exception as e:
    print(f'  (comparison failed: {e})')
"
fi

# Symlink latest run within the experiment folder
ln -sfn "$(basename "$LOG_DIR")" "${CONFIG_DIR}/runs/latest"

exit "$BENCH_EXIT"
