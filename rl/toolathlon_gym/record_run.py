"""Record process identity and source provenance for the watcher."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from decomposer.models import create_model


def profile_metadata(profile):
    model = create_model(profile)
    try:
        return {"profile": profile, "preserve_reasoning": model.preserve_reasoning,
                **model.model_dump(include={"model_name", "temperature", "top_p", "extra_body",
                                           "max_tokens", "max_retries"}, exclude_none=True)}
    finally:
        model.http_client.close()
        import asyncio
        asyncio.run(model.http_async_client.aclose())


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
                "subagent": profile_metadata("lmrouter/qwen_3_5_4b_unlooped_thinking"),
                "model_path": os.environ.get("MODEL_PATH"),
                "model_requested_path": os.environ.get("MODEL_CHECKPOINT_LINK"),
                "data_dir": os.environ.get("RL_DATA"),
                "config_name": os.environ.get("RL_CONFIG"),
                "trainer_module": os.environ.get("RL_TRAINER_MODULE"),
                "config_dir": os.environ.get("RL_CONFIG_DIR"),
                "nccl_transport": {key: os.environ.get(key) for key in ("NCCL_P2P_DISABLE", "NCCL_IB_DISABLE")},
                "checkpoint_path": str((args.directory / "checkpoints").resolve()),
                "ray_tmpdir": os.environ.get("RAY_TMPDIR"),
                "episode_timeout_seconds": float(os.environ.get("RL_EPISODE_TIMEOUT", "2700")),
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "policy_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "overrides": overrides}
    if os.environ.get("RL_CONFIG_DIR", "").endswith("/opd/toolathlon_gym"):
        metadata["teacher"] = profile_metadata("lmrouter/qwen_3_8_flash_next_non_thinking")
    (args.directory / "run.json").write_text(json.dumps(metadata, indent=2))
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", "rl/toolathlon_gym", "opd/toolathlon_gym",
                                    "gyms/toolathlon_gym", "src/decomposer"], text=True)
    (args.directory / "source.diff").write_text(diff)


if __name__ == "__main__":
    main()
