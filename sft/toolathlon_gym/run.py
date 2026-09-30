"""Resumable, coverage-first off-policy trace collection."""

import signal
import sys

from gyms.toolathlon_gym import run as gym
from . import collection


def main(argv=None):
    return collection.main(
        sys.argv[1:] if argv is None else argv,
        repo_root=gym.REPO_ROOT,
        toolathlon_root=gym.TOOLATHLON_ROOT,
        default_artifacts_dir=gym.REPO_ROOT / "artifacts/sft/toolathlon_gym",
        default_image=gym.DEFAULT_IMAGE,
        default_model=gym.DEFAULT_MODEL,
        default_subagent_model=gym.DEFAULT_SUBAGENT_MODEL,
        default_subagent_api_model=gym.DEFAULT_SUBAGENT_API_MODEL,
        default_subagent_port=gym.DEFAULT_SUBAGENT_PORT,
        start_vllm=gym.start_vllm,
        stop_vllm=gym.stop_vllm,
        docker=gym._docker,
    )


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, gym._handle_termination)
    main()
