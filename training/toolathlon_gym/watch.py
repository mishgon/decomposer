"""Read-only live RL status. Run on Hertz-2 with watch-rl.sh."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def show(root):
    runs = list(root.glob("*/run.json"))
    if not runs:
        print("RL setup in progress: no recorded trainer launch yet.")
        return
    run = max(runs, key=lambda path: path.stat().st_mtime)
    metadata = json.loads(run.read_text())
    try:
        fields = Path(f"/proc/{metadata['pid']}/stat").read_text().split()
        active = fields[21] == metadata["process_start_ticks"] and fields[2] != "Z"
    except FileNotFoundError:
        active = False
    elapsed = max(0, int(time.time() - metadata["started_at"]))
    print(f"TOOLATHLON RL  {run.parent.name}")
    print(f"Trainer: {'running' if active else 'stopped'} | PID {metadata['pid']} | "
          f"elapsed {elapsed // 3600}h {elapsed % 3600 // 60}m | policy GPU {metadata['policy_gpu']}")
    episodes = list((run.parent / "episodes").glob("*"))
    scored, rewards = 0, []
    for directory in episodes:
        result = directory / "evaluation.json"
        if result.exists():
            try:
                evaluation = json.loads(result.read_text())
                if "reward" in evaluation:
                    scored += 1
                    rewards.append(evaluation["reward"])
            except (ValueError, OSError):
                pass
    print(f"Episodes: {len(episodes)} started | {scored} scored")
    if rewards:
        print(f"All-episode native reward mean: {sum(rewards) / len(rewards):.3f} (not a validation metric)")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    directory = run.parent / "tensorboard"
    found = False
    if directory.exists():
        events = EventAccumulator(str(directory)).Reload()
        for tag in sorted(events.Tags()["scalars"]):
            if ("reward" in tag or "score" in tag) and ("mean" in tag or "val" in tag):
                point = events.Scalars(tag)[-1]
                print(f"step {point.step}: {tag} = {point.value:.4f}")
                found = True
    if not found:
        print("Training/validation metrics: not emitted yet. ETA: not established.")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True)
    print("GPU | memory MiB | utilization %\n" + gpu.stdout.strip())
    print(f"Artifacts: {run.parent}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "artifacts/training/toolathlon_gym"
    while True:
        if not args.once:
            print("\033[2J\033[H", end="")
        show(root)
        if args.once:
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
