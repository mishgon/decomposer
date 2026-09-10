#!/usr/bin/env python3
"""Render a veRL dashboard. Usage: .venv-rl/bin/python scripts/plot_rl.py [--run PATH].

Requires matplotlib and tensorboard. Defaults to the newest recorded Toolathlon
Gym RL run; writes plots/training_dashboard.png and plots/scalars.json there.
Safe to rerun while training: absent metrics are shown as not logged, never zero.
"""

import argparse
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "artifacts/training/toolathlon_gym"
PANELS = [
    ("Training reward", ["critic/rewards/mean"]),
    ("Evaluation reward (native score)", []),
    ("Policy entropy", ["actor/entropy"]),
    ("Response length (tokens)", ["response_length/mean"]),
    ("Agent turns", ["training/num_turns/mean"]),
    ("Gradient norm", ["actor/grad_norm"]),
    ("Mean advantage", ["critic/advantages/mean"]),
    ("Generation time (seconds)", ["timing_s/gen"]),
    ("PPO clip fraction", ["actor/pg_clipfrac", "actor/pg_clipfrac_lower"]),
    ("PPO KL estimate", ["actor/ppo_kl"]),
    ("Rollout / training divergence", ["rollout_corr/kl", "rollout_corr/k3_kl", "rollout_corr/chi2_token"]),
    ("Trajectory staleness", ["training/off_policy/trajectory_staleness/mean"]),
    ("Response clipping / abort fraction", ["response_length/clip_ratio", "response/aborted_ratio"]),
    ("Throughput (logged tokens/s)", ["perf/throughput"]),
    ("Timing (seconds)", ["timing_s/step", "timing_s/gen", "timing_s/update_actor"]),
    ("Learning rate", ["actor/lr"]),
    ("Actor loss", ["actor/loss"]),
    ("Actor GPU memory (GB)", ["actor/perf/max_memory_allocated_gb", "actor/perf/max_memory_reserved_gb"]),
]


def choose_run(root, explicit=None):
    if explicit:
        run = explicit if explicit.is_dir() else root / explicit
        if not run.is_dir():
            raise FileNotFoundError(f"Run directory not found: {explicit}")
        return run.resolve()
    records = list(root.glob("*/run.json"))
    if not records:
        raise FileNotFoundError(f"No recorded RL runs in {root}; use --root or --run")
    return max(records, key=lambda p: json.loads(p.read_text())["started_at"]).parent.resolve()


def read_scalars(directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    points = {}
    for parent in sorted({p.parent for p in directory.rglob("events.out.tfevents.*")}):
        events = EventAccumulator(str(parent), size_guidance={"scalars": 0}).Reload()
        for tag in events.Tags()["scalars"]:
            by_step = points.setdefault(tag, {})
            for event in events.Scalars(tag):
                if event.step not in by_step or event.wall_time >= by_step[event.step]["wall_time"]:
                    by_step[event.step] = {"step": event.step, "value": event.value,
                                          "wall_time": event.wall_time}
    return {tag: [values[s] for s in sorted(values)] for tag, values in sorted(points.items())}


def render(run, scalars, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure, axes = plt.subplots(6, 3, figsize=(15, 21), constrained_layout=True)
    evaluation = [tag for tag in scalars if re.fullmatch(r"val-core/.+/reward/mean@\d+", tag)]
    labels = ", ".join(tag.removeprefix("val-core/").split("/reward/")[0] for tag in evaluation)
    figure.suptitle(f"{run.name}\nEvaluation panels: {labels or 'not logged yet'}", fontsize=13)
    for index, (axis, (title, tags)) in enumerate(zip(axes.flat, PANELS)):
        tags = evaluation if index == 1 else tags
        plotted = 0
        for tag in tags:
            values = scalars.get(tag, [])
            if not values:
                continue
            axis.plot([p["step"] for p in values],
                      [p["value"] if math.isfinite(p["value"]) else float("nan") for p in values],
                      marker=".", linewidth=1, markersize=5, label=tag)
            plotted += 1
            if any(not math.isfinite(p["value"]) for p in values):
                axis.text(.02, .96, "Non-finite values logged", color="red", va="top", transform=axis.transAxes)
        if not plotted:
            axis.text(.5, .5, "Not logged yet / unavailable", ha="center", va="center",
                      color="gray", transform=axis.transAxes)
        elif plotted > 1 or index == 1:
            axis.legend(fontsize=6, loc="best")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Optimizer step")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        steps = [p["step"] for tag in tags for p in scalars.get(tag, [])]
        if steps and min(steps) == max(steps):
            axis.set_xlim(max(0, steps[0] - 1), steps[0] + 1)
        axis.grid(alpha=.2)
        axis.tick_params(labelsize=8)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "training_dashboard.png", dpi=160)
    plt.close(figure)
    # Null marks non-finite data in the portable JSON export; the chart flags it.
    export = {tag: [{**p, "value": p["value"] if math.isfinite(p["value"]) else None}
                    for p in values] for tag, values in scalars.items()}
    (output / "scalars.json").write_text(json.dumps({"run": str(run), "scalars": export}, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run", type=Path, help="Run directory or name under --root")
    parser.add_argument("--output", type=Path, help="Defaults to RUN/plots")
    args = parser.parse_args()
    run = choose_run(args.root, args.run)
    scalars = read_scalars(run / "tensorboard")
    output = args.output or run / "plots"
    render(run, scalars, output)
    print(f"Run: {run}\nScalar series: {len(scalars)}\nDashboard: {output / 'training_dashboard.png'}")
    if not scalars:
        print("No scalars logged yet. Rerun after baseline evaluation or an optimizer update finishes.")


if __name__ == "__main__":
    main()
