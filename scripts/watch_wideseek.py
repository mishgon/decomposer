"""Read-only WideSeek progress display; ETA covers every mode and repetition."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time


def duration(seconds):
    minutes = max(0, int(seconds)) // 60
    return f"{minutes // 60}h {minutes % 60:02d}m"


def display(root, run=None):
    queue_file = root / 'sequence.json'
    queue = json.loads(queue_file.read_text()) if queue_file.exists() and run is None else None
    if queue:
        run = queue.get('active') or (queue.get('completed') or [None])[-1]
    manifests = list(root.glob("*/manifest.json"))
    path = (root / run / "manifest.json") if run else max(
        manifests, key=lambda p: p.stat().st_mtime, default=None)
    if path is None or not path.exists():
        print(f"No run manifest under {root}")
        return
    manifest = json.loads(path.read_text())
    directory = path.parent
    settings = manifest["settings"]
    total = len(settings["tasks"]) * settings["repetitions"] * len(settings["modes"])
    rows = []
    for result in directory.glob("*/*/attempt-*/result.json"):
        try:
            rows.append(json.loads(result.read_text()))
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
    finished = done >= total
    end = max((r["finished_at"] for r in rows), default=now) if finished else now
    elapsed = max(1, end - manifest["started_at"])
    print(f"WIDESEEK  {directory.name}  ({datetime.now(timezone.utc):%H:%M:%S UTC})")
    print("-" * 76)
    if queue:
        print(f"Sequence: {queue['status']} | completed {len(queue['completed'])}/{len(queue['jobs'])}")
        print("Queued next: " + (', '.join(queue['pending']) or 'none'))
    print(f"Status: {'completed' if finished else 'running' if active else 'STOPPED / interrupted'} | concurrency {settings['concurrency']}")
    print(f"Elapsed: {duration(elapsed)} | Progress: {done}/{total} episodes ({100*done/total:.1f}%)")
    print(f"Tasks: {len(settings['tasks'])} | attempts per task/mode: {settings['repetitions']}")
    judge = settings['judge']
    print(f"Agent: {settings['model']} | Judge: {judge.get('model') if isinstance(judge, dict) else judge}")
    print()
    legacy = len(settings['modes']) > 1
    if legacy:
        print("Legacy combined run (new runs use one setup each)")
    for mode in settings['modes']:
        selected = [r for r in rows if r['mode'] == mode]
        scores = [r['evaluation'].get('score') for r in selected]
        mean = f"{sum(s or 0 for s in scores)/len(scores):.3f}" if scores else '--'
        print(f"Setup: {mode}" + (f" | {len(selected)}/{total//len(settings['modes'])} scored attempts" if legacy else ""))
        print(f"Quality: mean score {mean} | {sum(r['status']=='finished' for r in selected)} normal finishes | {sum(s is None for s in scores)} unscored")
    print("Mean score: native score; unscored counted as zero, NOT binary pass rate.")
    print("Outcomes:", dict(Counter(r['status'] for r in rows)))
    if finished:
        eta = '0h 00m — all scheduled episodes completed'
    elif not active:
        eta = '-- (no active run process holding its lock)'
    elif done < 10:
        eta = '-- (warming up; need 10 completed episodes)'
    else:
        remaining = (total-done) * elapsed / done
        finish = datetime.fromtimestamp(now+remaining, timezone.utc)
        eta = f"{duration(remaining)} | finish {finish:%Y-%m-%d %H:%M UTC} | {done*3600/elapsed:.1f} episodes/hour"
    print(f"ETA whole run: {eta}")
    if queue and queue['pending']:
        print("ETA covers the ACTIVE setup only; queued setup timing is not yet measured.")
    print("ETA uses observed wall throughput including judging; approximate, not a deadline.")
    if done:
        print(f"Last completion: {duration(now-max(r['finished_at'] for r in rows))} ago")
    print(f"Artifacts: {directory}")


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
