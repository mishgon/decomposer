#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
source training/rl/toolathlon_gym/router.sh
export MODEL_PATH="$(readlink -f "${MODEL_PATH:-$HOME/models/decomposer-4b-sft}")"
export PYTHONPATH="$PWD:$PWD/src:$PWD/external/verl${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$PWD/.venv-rl/bin:/usr/local/cuda/bin:$PATH"
export RL_GYM_IMAGE="$(podman image inspect --format '{{.Id}}' "${RL_GYM_IMAGE:-localhost/decomposer-toolathlon-rl:status-filter-fix}")"
export CUDA_VISIBLE_DEVICES="${POLICY_GPU:-2}"
export TOKENIZERS_PARALLELISM=false
ulimit -n 65536
: "${TEST_ROOT:?Set a fresh TEST_ROOT directory}"
mkdir "$TEST_ROOT"
server_pid=""
phase_pid=""
cleanup() {
    trap - EXIT TERM INT
    if [[ -n "$phase_pid" ]] && kill -0 "$phase_pid" 2>/dev/null; then
        kill -TERM "$phase_pid" 2>/dev/null || true
        for i in {1..30}; do kill -0 "$phase_pid" 2>/dev/null || break; sleep 1; done
        kill -KILL -- "-$phase_pid" 2>/dev/null || true
    fi
    if [[ -n "$server_pid" ]]; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for i in {1..15}; do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
        kill -KILL -- "-$server_pid" 2>/dev/null || true
    fi
    .venv-rl/bin/python -m training.rl.toolathlon_gym.throughput_cleanup "$TEST_ROOT"
}
trap cleanup EXIT
trap 'exit 130' TERM INT
setsid env -u VLLM_API_KEY .venv-rl/bin/vllm serve "$MODEL_PATH" \
    --served-model-name decomposer-4b-sft --host 127.0.0.1 --port 8026 \
    --max-model-len 16384 --max-num-seqs 256 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.45 --language-model-only --no-enable-prefix-caching \
    --enforce-eager --additional-config '{"gdn_prefill_backend":"triton"}' \
    > "$TEST_ROOT/server.log" 2>&1 &
server_pid=$!
ready=0
for i in {1..120}; do
    kill -0 "$server_pid" 2>/dev/null || { echo 'Inference server exited'; exit 1; }
    if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1; then ready=1; break; fi
    sleep 3
done
[[ "$ready" == 1 ]] || { echo 'Inference startup timed out'; exit 1; }
echo 'Running one-minute end-to-end rollout preflight'
setsid .venv-rl/bin/python -m training.rl.toolathlon_gym.throughput \
    --output "$TEST_ROOT/preflight" --concurrency 40 --limit 1 --timeout 60 --gpu "$CUDA_VISIBLE_DEVICES" \
    > "$TEST_ROOT/preflight.log" 2>&1 &
phase_pid=$!
wait "$phase_pid"
phase_pid=""
.venv-rl/bin/python -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["scored"]==1 and not s["infrastructure_errors"], s' "$TEST_ROOT/preflight/summary.json"
start=$SECONDS
read -r -a phases <<< "${THROUGHPUT_PHASES:-40 60 80 40-repeat}"
task_args=(--repetitions "${THROUGHPUT_REPETITIONS:-5}")
if [[ "${THROUGHPUT_ALL_TASKS:-0}" == 1 ]]; then task_args+=(--all-tasks); fi
for phase in "${phases[@]}"; do
    if [[ "$phase" == 40-repeat && $((SECONDS-start)) -gt 9900 ]]; then break; fi
    conc="${phase%%-*}"
    echo "Starting c$phase at $(date -u +%FT%TZ)"
    setsid .venv-rl/bin/python -m training.rl.toolathlon_gym.throughput \
        --output "$TEST_ROOT/c$phase" --concurrency "$conc" --gpu "$CUDA_VISIBLE_DEVICES" "${task_args[@]}" \
        > "$TEST_ROOT/c$phase.log" 2>&1 &
    phase_pid=$!
    wait "$phase_pid"
    phase_pid=""
    .venv-rl/bin/python -m training.rl.toolathlon_gym.throughput_cleanup "$TEST_ROOT/c$phase"
    .venv-rl/bin/python -c 'import json,sys; s=json.load(open(sys.argv[1])); print(json.dumps(s)); assert s["status"]=="complete" and s["infrastructure_errors"] <= s["concurrency"]*float(sys.argv[2]), "Too many infrastructure failures; stopping escalation"' "$TEST_ROOT/c$phase/summary.json" "${THROUGHPUT_MAX_ERROR_FRACTION:-0.1}"
done
