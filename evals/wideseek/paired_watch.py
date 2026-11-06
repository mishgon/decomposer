"""Publish/read paired-evaluation status through the existing ClearML connection."""
import argparse
import json
from pathlib import Path
import sys
import time


def render(snapshot):
    manifest = snapshot['manifest']
    done = len(manifest['episodes']) + len(manifest['errors'])
    total = manifest['attempts']
    elapsed = manifest.get('finished_at', time.time()) - manifest['started_at']
    age = max(0, time.time() - snapshot['updated_at'])
    eta = elapsed / done * (total - done) if done else None
    filled = int(30 * done / total)
    lines = [f"WIDESEEK  teacher vs student | {manifest['status'].upper()}",
             f"[{('#' * filled).ljust(30, '.')}] {done}/{total}  "
             f"conc={manifest['concurrency']}  elapsed={elapsed / 60:.1f}m  "
             + (f"rough ETA={eta / 60:.1f}m" if eta is not None else 'ETA=warming up'),
             f"Graph limits: lead={manifest['recursion_limit']} researcher={manifest['agent_recursion_limit']}  "
             f"snapshot age={age:.0f}s" + ('  STALE' if age > 90 and manifest['status'] == 'running' else ''),
             '', 'MODEL     SPLIT     DONE    ITEM-F1   UNSCORED   MEDIAN AGENT']
    for model in ('student', 'teacher'):
        for split, count in [('train', 16), ('heldout', 8)]:
            row = snapshot.get('summary', {}).get(model, {}).get(split, {})
            score = row.get('mean_native_score')
            seconds = row.get('median_agent_seconds')
            score_text = f'{score:.3f}' if score is not None else '--'
            seconds_text = f'{seconds:.0f}s' if seconds is not None else '--'
            lines.append(f"{model:<9} {split:<8} {row.get('attempts', 0):>2}/{count:<2}   "
                         f"{score_text:>7}   {row.get('unscored', 0):>8}   {seconds_text:>12}")
    lines += ['', f"Runner exceptions: {len(manifest['errors'])}. Scores are partial until both models finish.",
              'Refresh: 15s | Ctrl-C closes the watcher only.']
    return '\n'.join(lines)


def publish(root):
    from clearml import Task
    task = Task.create(project_name='decomposer/wideseek', task_name=f'{root.name}-watch',
                       task_type='testing', detect_repository=False)
    task.mark_started()
    (root / 'watch-task-id').write_text(task.id + '\n')
    print(task.id, flush=True)
    while True:
        manifest = json.loads((root / 'manifest.json').read_text())
        path = root / 'summary.json'
        snapshot = {'updated_at': time.time(), 'manifest': manifest,
                    'summary': json.loads(path.read_text()) if path.exists() else {}}
        task.set_configuration_object('watch', config_text=json.dumps(snapshot), config_type='json')
        if manifest['status'] != 'running':
            task.mark_completed()
            return
        time.sleep(15)


def watch(task_id, once):
    print('Connecting to WideSeek evaluation status…', flush=True)
    from clearml import Task
    task = Task.get_task(task_id=task_id)
    while True:
        try:
            task.reload()
            snapshot = json.loads(task.get_configuration_object('watch') or '{}')
            text = render(snapshot) if snapshot else 'Waiting for first status snapshot…'
            if sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print(text, flush=True)
        except Exception as exc:
            print(f'Status refresh failed: {type(exc).__name__}: {exc}', flush=True)
            if once:
                raise
        if once:
            return
        time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--publish', type=Path)
    source.add_argument('--task-id')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    try:
        publish(args.publish) if args.publish else watch(args.task_id, args.once)
    except KeyboardInterrupt:
        pass
