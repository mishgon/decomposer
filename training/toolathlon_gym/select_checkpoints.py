"""Label last and highest-reward evaluated RL checkpoints without copying weights."""

import argparse
import json
from pathlib import Path


def select(directory, panels):
    checkpoints = {int(p.name.removeprefix("global_step_")): p
                   for p in directory.glob("global_step_*")
                   if (p / "actor").is_dir()}
    if not checkpoints:
        return {}
    eligible = [p for p in panels if p["complete"] and p["step"] in checkpoints
                and p["panel"] == "toolathlon_gym/train_probe"]
    choices = {"last": max(checkpoints)}
    if eligible:
        best = max(eligible, key=lambda p: (p["reward"], p["step"]))
        choices["best"] = best["step"]
    for name, step in choices.items():
        link = directory / name
        if link.exists() and not link.is_symlink():
            raise ValueError(f"Refusing to replace non-symlink {link}")
        temporary = directory / f".{name}.next"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(checkpoints[step].name, target_is_directory=True)
        temporary.replace(link)
    metadata = {"selection": "Highest complete mean native reward on the fixed training-task evaluation panel",
                "held_out": False, "steps": choices, "evaluated_checkpoints": eligible}
    (directory / "selection.json").write_text(json.dumps(metadata, indent=2))
    return choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = args.run / "reward_report.json"
    print(select(args.run / "checkpoints", json.loads(report.read_text()) if report.exists() else []))


if __name__ == "__main__":
    main()
