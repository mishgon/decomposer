#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:-}" in
    full|smoke) profile="$1"; shift ;;
    *) echo "Usage: $0 {full|smoke} [Hydra overrides...]" >&2; exit 2 ;;
esac
: "${RL_DATA:?Set RL_DATA to a new OPD dataset directory}"
: "${RL_ARTIFACTS:?Set RL_ARTIFACTS to a new OPD run directory}"
test ! -e "$RL_ARTIFACTS/run.json"
patch_file="$PWD/opd/toolathlon_gym/patches/verl-hosted-teacher.patch"
# Setup applies this opt-in hook; it does not alter RL's objective.
if ! git -C external/verl apply --recount --reverse --check "$patch_file" 2>/dev/null; then
    echo "Apply $patch_file to this checkout's external/verl before launching" >&2
    exit 1
fi
if [ ! -e "$RL_DATA" ]; then
    .venv-rl/bin/python -m rl.toolathlon_gym.task_profiles \
        --prepare "$RL_DATA" --profile "$profile" --groups-per-task 1
fi
export RL_CONFIG_DIR="$PWD/opd/toolathlon_gym"
exec bash rl/toolathlon_gym/train.sh "$profile" "$@"
