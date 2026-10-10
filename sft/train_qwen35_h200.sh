#!/usr/bin/env bash
# Qwen3.5 SFT on Hertz-2's H200s with the environment from
# docs/sft_qwen35_h200_benchmark.md (Environment).
#
#   sft/train_qwen35_h200.sh <config> <gpus, e.g. 4,5,6,7> [sft.train arguments]
#
# It refuses GPUs that hold memory: other users' jobs do not always show up as
# compute processes.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <config> <gpus> [sft.train arguments]" >&2
  exit 2
fi
config=$1
gpus=$2
shift 2

busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
  awk -F', ' -v wanted="$gpus" 'BEGIN { split(wanted, list, ","); for (i in list) want[list[i]] = 1 }
    ($1 in want) && $2 > 1024 { printf "%s%s (%s MiB)", sep, $1, $2; sep = ", " }')
if [[ -n $busy ]]; then
  echo "GPUs in use: $busy" >&2
  exit 1
fi

repo=$(cd "$(dirname "$0")/.." && pwd)
artifacts=/mnt/share14T-2/sukhorukov/decomposer_artifacts
# Triton 3.7.1 and the sm_90 causal-conv1d build: FLA refuses the venv's Triton on
# Hopper, and without causal-conv1d Transformers silently uses the slow convolution.
bundle=$artifacts/kernels/sft/qwen35-hf-fa2-fla-h200-v1
snapshots=$HOME/.cache/huggingface/hub
export PYTHONPATH=$bundle/triton:$bundle/causal${PYTHONPATH:+:$PYTHONPATH}
# Pinned FA2 and FA3 kernels from local snapshots; per-rank Hub listings hit rate limits.
export LOCAL_KERNELS=kernels-community/flash-attn2=$snapshots/kernels--kernels-community--flash-attn2/snapshots/c269cc539ad0c1fc0899abd4b05ecc1303d6c4b1:kernels-community/flash-attn3=$snapshots/kernels--kernels-community--flash-attn3/snapshots/3c1451f803c146c54222305a8c350f7f46bb5135
# Keep FLA's autotuning results across processes.
export TRITON_CACHE_DIR=$artifacts/training/sft/benchmarks/qwen35-4b-unloop-h200-20261007/triton-cache
export TRITON_CACHE_AUTOTUNING=1
export FLA_TILELANG=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export CLEARML_CONFIG_FILE=${CLEARML_CONFIG_FILE:-$HOME/.secrets/clearml.conf}
export CUDA_VISIBLE_DEVICES=$gpus

cd "$repo"
exec .venv/bin/torchrun --standalone --nproc-per-node="$(awk -F, '{ print NF }' <<<"$gpus")" \
  -m sft.train --config "$config" "$@"
