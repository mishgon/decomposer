"""Live comparison of hosted synthetic evaluations; reads local artifacts only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600}h {seconds // 60 % 60:02d}m {seconds % 60:02d}s'


def spark(values):
    # Native scores have a fixed 0..1 scale, shared by both models.
    return ''.join('▁▂▃▄▅▆▇█'[round(7 * max(0, min(1, v)))] for v in values[-24:]) or '--'


def show(root, title="SYNTH EVAL"):
    print(f'{title}  |  {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}')
    print('Same held-out tasks | scores below cover completed attempts only')
    print('=' * 86)
    launches = sorted(root.glob('*-launch.json'))
    if not launches:
        print(f'Waiting for evaluation launches under {root}')
    for launch_path in launches:
        launch = read(launch_path)
        label = launch_path.name.removesuffix('-launch.json')
        directory = root / label
        manifest = read(directory / 'manifest.json')
        rows = manifest.get('episodes', [])
        total = manifest.get('tasks', 0) * manifest.get('repetitions', 0)
        state = manifest.get('status', 'starting')
        if state in {'running', 'starting'}:
            try:
                command = Path(f"/proc/{launch['pid']}/cmdline").read_bytes()
                if str(directory).encode() not in command:
                    state = 'STOPPED (stale manifest)'
            except (OSError, KeyError):
                state = 'STOPPED (no evaluator process)'
        elapsed = (manifest.get('finished_at') or time.time()) - manifest.get('started_at', launch.get('started_at', time.time()))
        done = len(rows)
        progress = round(24 * done / total) if total else 0
        print(f"\n{label.upper()}  |  {state}  |  concurrency {manifest.get('concurrency', '?')}")
        print(f"Model: {manifest.get('model', launch.get('model', '?'))}")
        print(f"[{'█' * progress}{'░' * (24-progress)}] {done}/{total or '?'}  |  elapsed {duration(elapsed)}")
        if state == 'completed':
            eta = 'finished'
        elif state == 'running' and 'selection_progress' in manifest:
            eta = 'stops when the 20-task target is reached; full-pool ETA does not apply'
        elif state != 'running':
            eta = '-- (not running)'
        elif done >= 3 and total:
            eta = '~' + duration((total-done) * elapsed / done) + ' (observed wall throughput)'
        else:
            eta = '-- (warming up; need 3 completed attempts)'
        print(f'ETA: {eta}')
        if 'selection_progress' in manifest:
            selection = manifest['selection_progress']
            if selection['status'] == 'selected':
                print('Panel ready: teacher 20/20, student 2/20 (selection runs)')
            else:
                print(f"Panel target: teacher-only passes {selection['teacher_only_passes']}/18"
                      f" | shared passes {selection['both_pass']}/2")
        if rows:
            errors = Counter(r.get('error_type') or ('cleanup' if r.get('cleanup_errors') else 'other')
                             for r in rows if r['status'] != 'finished')
            print(f"Mean score: {sum(r['score'] for r in rows)/done:.3f}  |  "
                  f"Passed: {sum(r['passed'] for r in rows)}/{done}  |  Errors: {sum(errors.values())}/{done}")
            print('Score trend: ' + spark([r['score'] for r in rows]) + '  (last 24 completions, fixed 0..1)')
            print('Errors: ' + (', '.join(f'{k}={v}' for k,v in sorted(errors.items())) or 'none'))
        else:
            print('Mean score: --  |  Passed: --  |  Errors: --')
        if manifest.get('benchmark') != 'decomposer-demo':
            unfinished = sum(not p.with_name('result.json').exists() for p in directory.glob('*/task.json'))
            print(f'Unfinished episodes on disk: {unfinished} (can include interrupted attempts)')
    print(f'\nArtifacts: {root}')
    print('Read-only | Ctrl-C exits this dashboard; evaluations continue.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--title', default='SYNTH EVAL')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    try:
        while True:
            if not args.once and sys.stdout.isatty():
                print('\033[2J\033[H', end='', flush=True)
            show(args.root, args.title)
            if args.once:
                break
            time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
