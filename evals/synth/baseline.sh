#!/usr/bin/env bash
# Linux/Hertz-2: own one inference process and release it after the baseline.
set -euo pipefail
cd "$(dirname "$0")/../.."
gpu="${1:?GPU index}"
checkpoint="${2:?Stock Qwen3.5-4B checkpoint directory}"
output="${3:?New baseline output directory}"
test -f "$checkpoint/config.json"
test ! -e "$output"
if [[ -n "$(ss -ltnH 'sport = :8024')" ]]; then
    echo "Port 8024 is already occupied; refusing to reuse another server." >&2
    exit 1
fi
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used > 1024 )); then
    echo "GPU $gpu is occupied ($used MiB); refusing to launch." >&2
    exit 1
fi
export PYTHONPATH="$PWD/src:$PWD"
mkdir -p "${output}.logs"
CUDA_VISIBLE_DEVICES="$gpu" setsid python -m vllm.entrypoints.cli.main serve "$checkpoint" \
    --served-model-name Qwen/Qwen3.5-4B --host 127.0.0.1 --port 8024 \
    --max-model-len 32768 --gpu-memory-utilization 0.45 --language-model-only \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --enable-prefix-caching --enforce-eager --additional-config '{"gdn_prefill_backend":"triton"}' \
    > "${output}.logs/vllm.log" 2>&1 &
server_pid=$!
cleanup() {
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    for ((i=0; i<30; i++)); do
        kill -0 -- "-$server_pid" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready=0
for ((i=0; i<300; i++)); do
    kill -0 "$server_pid" 2>/dev/null || { echo "vLLM failed; see ${output}.logs/vllm.log" >&2; exit 1; }
    if curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:8024/health >/dev/null; then ready=1; break; fi
    sleep 2
done
(( ready == 1 )) || { echo "vLLM readiness timed out" >&2; exit 1; }
python -u -m evals.synth.run --split eval --tasks 8 -n 4 --concurrency 8 --output "$output" \
    > "${output}.logs/eval.log" 2>&1
