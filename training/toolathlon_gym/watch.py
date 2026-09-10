"""Read-only live RL status. Run on Hertz-2 with watch-rl.sh."""

import argparse
import json
import math
import subprocess
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from statistics import mean

from training.toolathlon_gym.report import summarize


def duration(seconds):
    minutes = max(0, round(seconds / 60))
    return f"{minutes // 60}h {minutes % 60:02d}m"


def trend(values):
    values = [v for v in values[-8:] if math.isfinite(v)]
    if not values:
        return "--"
    low, high = min(values), max(values)
    bars = "▁▂▃▄▅▆▇█"
    return "".join(bars[round(7 * (v - low) / (high - low))] if high > low else "▄" for v in values)


@lru_cache(maxsize=512)
def episode_record(path, modified):
    # Cache only the summary, not the potentially large model histories.
    trace = json.loads(path.read_text())
    result = json.loads(path.with_name("evaluation.json").read_text())
    return ({key: trace.get(key) for key in ("task_id", "data_source", "weight_versions")}, result,
            trace["stop_reason"], trace["elapsed_seconds"], modified)


@lru_cache(maxsize=8)
def expected_panels(directory):
    import pandas as pd
    frame = pd.read_parquet(Path(directory) / "evaluation.parquet")
    return {source: {row["task_id"] for row in group.extra_info}
            for source, group in frame.groupby("data_source")}


def planned_evaluations(total, frequency, before):
    rounds = {0} if before else set()
    if frequency > 0:
        rounds.update(range(frequency, total + 1, frequency))
        rounds.add(total)
    return rounds


def estimate_eta(elapsed, scored, total_episodes, steps_left, update_times, validation_left, validation_times):
    if update_times and (not validation_left or validation_times):
        seconds = steps_left * mean(update_times[-4:])
        seconds += validation_left * (mean(validation_times) if validation_times else 0)
        return seconds, "measured update + validation timings; approximate"
    if scored >= 3:
        return max(0, total_episodes - scored) * elapsed / scored, "LOW confidence: episode-throughput extrapolation"
    return None, "warming up; not enough completed episodes"


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
    import yaml
    config_path = run.parent / "hydra/.hydra/config.yaml"
    if not config_path.exists():
        print(f"TOOLATHLON RL  {run.parent.name} | {'starting' if active else 'stopped'} | {duration(elapsed)}")
        return
    config = yaml.safe_load(config_path.read_text())
    trainer = config["trainer"]
    rollout = config["actor_rollout_ref"]["rollout"]
    total = trainer["total_training_steps"]
    samples = rollout["val_kwargs"]["n"]
    data_dir = metadata.get("data_dir")
    if not data_dir and active:
        # Read only the one non-secret setting needed for the task manifest.
        try:
            environment = Path(f"/proc/{metadata['pid']}/environ").read_bytes()
        except OSError:
            environment = b""
        for entry in environment.split(b"\0"):
            if entry.startswith(b"RL_DATA="):
                data_dir = entry.partition(b"=")[2].decode()
                break
    if not data_dir:
        # Older launches did not record this field; the final report remains readable.
        data_dir = config["data"]["train_files"].rsplit("/", 1)[0]
    expected = expected_panels(data_dir) if Path(data_dir).is_dir() else {}
    episodes = list((run.parent / "episodes").glob("*"))
    scored, records = 0, []
    for directory in episodes:
        result = directory / "evaluation.json"
        if result.exists():
            try:
                evaluation = json.loads(result.read_text())
                if "reward" in evaluation:
                    scored += 1
                trace = directory / "trace.json"
                if "reward" in evaluation and trace.exists():
                    records.append(episode_record(trace, trace.stat().st_mtime))
            except (ValueError, OSError):
                pass
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    directory = run.parent / "tensorboard"
    scalars = {}
    if directory.exists():
        events = EventAccumulator(str(directory)).Reload()
        scalars = {tag: events.Scalars(tag) for tag in events.Tags()["scalars"]}
    if not active:
        log = run.parent / "trainer.log"
        end = max([metadata["started_at"], log.stat().st_mtime if log.exists() else 0,
                   *(r[4] for r in records), *(p.wall_time for points in scalars.values() for p in points)])
        elapsed = max(0, int(end - metadata["started_at"]))
    def latest(tag):
        return scalars[tag][-1].value if scalars.get(tag) else None
    step = int(latest("training/global_step") or 0)
    panels = summarize([(r[0], r[1]) for r in records], expected, samples) if expected else []
    complete = {p["step"] for p in panels if all(
        any(q["step"] == p["step"] and q["panel"] == source and q["complete"] for q in panels)
        for source in expected)}
    step = max([step, *complete])
    rounds = planned_evaluations(total, trainer["test_freq"], trainer["val_before_train"])
    validation_size = sum(map(len, expected.values())) * samples
    total_episodes = total * config["data"]["train_batch_size"] * rollout["n"] + len(rounds) * validation_size
    update_times = []
    for point in scalars.get("timing_s/gen", []):
        value = point.value
        for tag in ("old_log_prob", "adv", "update_actor", "save_checkpoint", "update_weights"):
            value += sum(p.value for p in scalars.get(f"timing_s/{tag}", []) if p.step == point.step)
        update_times.append(value)
    validation_times = []
    for version in complete:
        rows = [r for r in records if r[0]["data_source"] in expected
                and r[0]["weight_versions"]["min_global_steps"] == version]
        validation_times.append(max(r[4] for r in rows) - min(r[4] - r[3] for r in rows))
    eta, basis = estimate_eta(elapsed, scored, total_episodes, max(0, total - step), update_times,
                              len(rounds - complete), validation_times)
    grad, clip = latest("actor/grad_norm"), latest("actor/pg_clipfrac")
    done = step >= total and rounds <= complete
    nonfinite = any(v is not None and not math.isfinite(v) for v in
                    (grad, clip, latest("actor/loss")))
    health = "COOKED: non-finite training metric" if nonfinite else (
        "DONE" if done else "STOPPED early — inspect trainer.log" if not active else "COOKING: process alive")
    if active and not nonfinite:
        if grad == 0:
            health = "WATCH: zero gradient on latest update (equal rewards can cause this)"
        if clip is not None and clip > .5:
            health = "WATCH: >50% of policy updates clipped"
    baseline = 0 in rounds and 0 not in complete
    phase = f"baseline ({scored}/{validation_size} scored)" if baseline else "training / periodic validation"
    if "overfit" in trainer.get("experiment_name", ""):
        phase = "single-task overfit (NO holdout)"
    if done:
        phase = "complete"
    print(f"TOOLATHLON RL  {run.parent.name} | {health}")
    print(f"Elapsed {duration(elapsed)} | phase: {phase} | updates {step}/{total}")
    print(f"Episodes {scored}/{total_episodes or '?'} scored | {len(episodes) - scored} unfinished")
    if done:
        print("ETA: finished")
    elif active and expected:
        print(f"ETA whole scheduled pilot: {duration(eta) if eta is not None else '--'} | {basis}")
    else:
        print("ETA: -- (trainer stopped or task manifest unavailable)")
    print("Reward (native partial score, NOT pass rate):")
    print("  Fixed panel             Base    Latest    Delta")
    for source in sorted(expected):
        rows = sorted([p for p in panels if p["panel"] == source and p["complete"]], key=lambda p: p["step"])
        base = next((p["reward"] for p in rows if p["step"] == 0), None)
        after = rows[-1] if rows and rows[-1]["step"] > 0 else None
        b = f"{base:.3f}" if base is not None else "--"
        a = f"{after['reward']:.3f} s{after['step']}" if after else "--"
        delta = f"{after['reward'] - base:+.3f}" if after and base is not None else "--"
        print(f"  {source.rsplit('/', 1)[-1]:<22} {b:>5}  {a:>10}  {delta:>7}")
    rewards = [p.value for p in scalars.get("critic/rewards/mean", [])]
    if rewards:
        print(f"Batch reward: {trend(rewards)} {rewards[-1]:.3f} (varying tasks; not proof of learning)")
    if grad is None:
        print("Training health: awaiting first optimizer update; learning not established")
    else:
        print(f"Training health: grad {grad:.3g} | clip {clip:.1%}" if clip is not None else f"Training health: grad {grad:.3g}")
    stops = Counter(r[2] for r in records)
    print(f"Outcomes: {stops['finished']} finished | {stops['TimeoutError']} timeouts | "
          f"{sum(v for k, v in stops.items() if k not in ('finished', 'TimeoutError'))} other stops")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True)
    used = [line for line in gpu.stdout.splitlines() if int(line.split(',')[1].strip().split()[0]) > 128]
    print("Host GPU | memory MiB | utilization %\n" + "\n".join(used))
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
