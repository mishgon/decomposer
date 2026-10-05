"""Read-only WideSeek progress display; ETA covers every mode and repetition."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import json
from pathlib import Path
import statistics
import time

from sft.wideseek.scheduler import qualifies
from gyms.wideseek.metrics import subagent_counts


def duration(seconds):
    minutes = max(0, int(seconds)) // 60
    return f"{minutes // 60}h {minutes % 60:02d}m"


@lru_cache(maxsize=4096)
def trace_counts(path):
    return subagent_counts(json.loads(path.read_text())['messages'])


def display(root, run=None):
    roots = [root, *(root / stage / "wideseek/runs" for stage in ("gyms", "evals", "sft"))]
    manifests = [p for base in roots for p in base.glob("*/manifest.json")]
    path = next((base / run / "manifest.json" for base in roots
                 if (base / run / "manifest.json").exists()), None) if run else max(
        manifests, key=lambda p: p.stat().st_mtime, default=None)
    if path is None or not path.exists():
        print(f"No run manifest under {root}")
        return
    manifest = json.loads(path.read_text())
    directory = path.parent.resolve()
    settings = manifest["settings"]
    scheduler_path = directory / 'scheduler.json'
    scheduler = json.loads(scheduler_path.read_text()) if scheduler_path.exists() else None
    total = len(settings["tasks"]) * settings["repetitions"] * len(settings["modes"])
    rows = []
    counts = {}
    for result in directory.glob("*/*/attempt-*/result.json"):
        try:
            row = json.loads(result.read_text())
            row['trace_available'] = (result.parent / row.get('execution_directory', '') / 'trace.json').is_file()
            rows.append(row)
            if 'subagent_statistics' in row:
                counts[id(row)] = row['subagent_statistics']
            elif row['mode'] == 'simple':
                counts[id(row)] = (0, 0)
            else:
                trace = result.parent / row.get('execution_directory', '') / 'trace.json'
                try:
                    counts[id(row)] = trace_counts(trace)
                except (OSError, ValueError, KeyError):
                    pass
        except (OSError, ValueError):
            pass  # A result may be in the middle of being written.
    active = False
    lock = directory.parent / (directory.name + ".lock")
    if lock.exists():
        with lock.open("r") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                active = True
    now = time.time()
    done = len(rows)
    ended = {(r['task_id'], r['attempt']) for r in rows}
    wave = scheduler['waves'][-1]['jobs'] if scheduler and scheduler['waves'] else []
    pending = [job for job in wave if tuple(job) not in ended]
    finished = scheduler['phase'] == 'complete' if scheduler else done >= total
    end = max((r["finished_at"] for r in rows), default=now) if finished else now
    elapsed = max(1, end - manifest["started_at"])
    print(f"WIDESEEK  {directory.name}  ({datetime.now(timezone.utc):%H:%M:%S UTC})")
    print("-" * 76)
    print(f"Status: {'completed' if finished else 'running' if active else 'STOPPED / interrupted'} | concurrency {settings['concurrency']}")
    if scheduler:
        covered = {r['task_id'] for r in rows if qualifies(r, scheduler['success_threshold'])}
        culled = len(scheduler['culled_tasks'])
        print(f"Elapsed: {duration(elapsed)} | Attempts ended: {done}")
        print(f"Tasks: {len(settings['tasks'])} | coverage {len(covered)}/{len(settings['tasks'])} | exhausted {culled}")
        print(f"Phase: {scheduler['phase']} | wave {len(scheduler['waves'])}: {len(wave)-len(pending)}/{len(wave)} ended | goal {scheduler['target_successes']} traces/task")
    else:
        print(f"Elapsed: {duration(elapsed)} | Attempts ended: {done}/{total} ({100*done/total:.1f}%)")
        print(f"Tasks: {len(settings['tasks'])} | attempts per task/mode: {settings['repetitions']}")
    judge = settings['judge']
    print(f"Agent: {settings['model']} | Judge: {judge.get('model') if isinstance(judge, dict) else judge}")
    if 'decomposer' in settings['modes'] and 'model_profiles' in settings:
        profile = settings['model_profiles']['subagent']
        print(f"Subagents: {profile['model_name']} | preserve reasoning: {profile['preserve_reasoning']}")
    print()
    legacy = len(settings['modes']) > 1
    if legacy:
        print("Legacy combined run (new runs use one setup each)")
    for mode in settings['modes']:
        selected = [r for r in rows if r['mode'] == mode]
        scores = [r['evaluation'].get('score') for r in selected]
        mean = f"{sum(s or 0 for s in scores)/len(scores):.3f}" if scores else '--'
        returned = sum(r['status'] == 'finished' for r in selected)
        print(f"Setup: {mode}")
        print(f"Attempts ended: {len(selected)} = {returned} returned an answer + {len(selected)-returned} stopped early")
        stops = Counter(r['status'] for r in selected if r['status'] != 'finished')
        if stops:
            print("Stop reasons: " + ', '.join(f"{name.replace('_', ' ')}: {count}" for name, count in sorted(stops.items())))
        print(f"Mean native score: {mean} across all {len(selected)} ended attempts (not pass rate)")
        print(f"Evaluation errors: {sum(s is None for s in scores)} | Missing answers and evaluation errors count as zero in mean")
        samples = [counts[id(r)] for r in selected if id(r) in counts]
        if samples:
            print(f"Subagents/attempt: {sum(s[0] for s in samples)/len(samples):.2f} | Peak unawaited/attempt: {sum(s[1] for s in samples)/len(samples):.2f} (mean over {len(samples)} ended attempts)")
        else:
            print("Subagents/attempt: -- | Peak unawaited/attempt: --")
    if scheduler:
        seconds = [r['finished_at']-r['started_at'] for r in rows]
        if finished:
            eta = '0h 00m — collection budgets/targets exhausted'
        elif not active:
            eta = f"-- ({scheduler.get('status', 'interrupted')}; no active collector)"
        elif seconds and pending:
            eta = f"~{duration(len(pending)*statistics.mean(seconds)/settings['concurrency'])} | episode-time estimate"
        else:
            eta = '-- (warming up)'
        print(f"ETA current wave: {eta}")
        if not finished:
            print("ETA whole collection: -- (later waves depend on scores)")
    elif finished:
        eta = '0h 00m — all scheduled episodes completed'
    elif not active:
        eta = '-- (no active run process holding its lock)'
    elif done < 10:
        eta = '-- (warming up; need 10 completed episodes)'
    else:
        remaining = (total-done) * elapsed / done
        finish = datetime.fromtimestamp(now+remaining, timezone.utc)
        eta = f"{duration(remaining)} | finish {finish:%Y-%m-%d %H:%M UTC} | {done*3600/elapsed:.1f} episodes/hour"
    if not scheduler:
        print(f"ETA whole run: {eta}")
        print("ETA uses observed wall throughput including judging; approximate, not a deadline.")
    if done:
        print(f"Last completion: {duration(now-max(r['finished_at'] for r in rows))} ago")
    print(f"Artifacts: {directory}")
    if (directory / 'collection.json').exists():
        collection = json.loads((directory / 'collection.json').read_text())
        if scheduler:
            print(f"Indexed traces (last completed wave): {collection['successful_traces']}")
        else:
            print(f"Traces: {collection['successful_traces']} successful | coverage {collection['covered_tasks']}/{collection['tasks']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('run', nargs='?')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=15)
    args = parser.parse_args()
    while True:
        if not args.once:
            print('\033[2J\033[H', end='', flush=True)
        display(args.root, args.run)
        if args.once:
            break
        time.sleep(max(1, args.interval))
