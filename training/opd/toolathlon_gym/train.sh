#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
case "${1:-}" in
    full|smoke) profile="$1"; shift ;;
    *) echo "Usage: $0 {full|smoke} [Hydra overrides...]" >&2; exit 2 ;;
esac
: "${RL_DATA:?Set RL_DATA to a new OPD dataset directory}"
: "${RL_ARTIFACTS:?Set RL_ARTIFACTS to a new OPD run directory}"
test ! -e "$RL_ARTIFACTS/run.json"
export OPD_TEACHER_MODEL="${OPD_TEACHER_MODEL:-Qwen/Qwen3.8-Flash-Next-NVFP4}"
export OPD_TEACHER_URL="${OPD_TEACHER_URL:-https://lmrouter.2a2i.org/v1}"
export OPD_TEACHER_HOST="${OPD_TEACHER_HOST:-lmrouter.2a2i.org:176.108.242.226}"
export SUBAGENT_MODEL=Qwen/Qwen3.5-4B
export SUBAGENT_URL="${SUBAGENT_URL:-https://lmrouter.2a2i.org/v1}"
export SUBAGENT_HOST="${SUBAGENT_HOST:-lmrouter.2a2i.org:176.108.242.226}"
source training/rl/toolathlon_gym/router.sh
export OPD_TEACHER_API_KEY="${OPD_TEACHER_API_KEY:-$VLLM_API_KEY}"
patch_file="$PWD/training/opd/toolathlon_gym/patches/verl-hosted-teacher.patch"
# Apply in the OPD checkout, never in the active RL checkout.
if ! git -C external/verl apply --recount --reverse --check "$patch_file" 2>/dev/null; then
    echo "Apply $patch_file to this checkout's external/verl before launching" >&2
    exit 1
fi
if [ ! -e "$RL_DATA" ]; then
    .venv-rl/bin/python -m training.rl.toolathlon_gym.task_profiles \
        --prepare "$RL_DATA" --profile "$profile" --groups-per-task 1
fi
export RL_CONFIG_DIR="$PWD/training/opd/toolathlon_gym"
exec bash training/rl/toolathlon_gym/train.sh "$profile" "$@"
