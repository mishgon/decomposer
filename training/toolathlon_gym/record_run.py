"""Record process identity and source provenance for the watcher."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    ticks = Path(f"/proc/{args.pid}/stat").read_text().split()[21]
    age = float(Path("/proc/uptime").read_text().split()[0]) - int(ticks) / os.sysconf("SC_CLK_TCK")
    metadata = {"pid": args.pid, "started_at": time.time() - age,
                "process_start_ticks": ticks,
                "gym_image": os.environ.get("RL_GYM_IMAGE"),
                "subagent_url": os.environ.get("SUBAGENT_URL"),
                "subagent_model": os.environ.get("SUBAGENT_MODEL"),
                "subagent_host": os.environ.get("SUBAGENT_HOST"),
                "model_path": os.environ.get("MODEL_PATH"),
                "model_requested_path": os.environ.get("MODEL_CHECKPOINT_LINK"),
                "data_dir": os.environ.get("RL_DATA"),
                "checkpoint_path": str((args.directory / "checkpoints").resolve()),
                "episode_timeout_seconds": float(os.environ.get("RL_EPISODE_TIMEOUT", "2700")),
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "policy_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "overrides": overrides}
    (args.directory / "run.json").write_text(json.dumps(metadata, indent=2))
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", "training/toolathlon_gym",
                                    "gyms/toolathlon_gym", "src/decomposer"], text=True)
    (args.directory / "source.diff").write_text(diff)


if __name__ == "__main__":
    main()
