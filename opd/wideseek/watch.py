"""Read-only WideSeek OPD dashboard for bare metal and MLSpace."""
import argparse
import ast
from datetime import datetime, timezone
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time


def duration(seconds):
    minutes = max(0, round(seconds / 60))
    return f'{minutes // 60}h {minutes % 60:02d}m'


def trend(values):
    values = [v for v in values[-8:] if math.isfinite(v)]
    if not values:
        return '--'
    low, high = min(values), max(values)
    return ''.join('▁▂▃▄▅▆▇█'[round(7 * (v-low)/(high-low))] if high > low else '▄'
                   for v in values)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}  # Writers may be replacing a manifest during this refresh.


def job_status(name):
    try:
        result = subprocess.run(['/home/jovyan/.local/bin/mls', 'job', 'status', name],
                                capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            return ast.literal_eval(result.stdout.strip())
    except (OSError, ValueError, SyntaxError, subprocess.TimeoutExpired):
        pass
    return {'status': 'unknown (MLSpace status unavailable)'}


def scalars(directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    series = {}
    # veRL may put the event file inside an experiment subdirectory.
    for parent in sorted({p.parent for p in directory.rglob('events.out.tfevents.*')}):
        events = EventAccumulator(str(parent), size_guidance={'scalars': 0}).Reload()
        for tag in events.Tags()['scalars']:
            series.setdefault(tag, []).extend(events.Scalars(tag))
    for tag, points in series.items():
        series[tag] = sorted(points, key=lambda p: (p.step, p.wall_time))
    return series


def update_eta(points, step, total):
    by_step = {p.step: p.wall_time for p in points}
    recent = sorted(by_step)[-5:]
    if total is None or len(recent) < 2 or recent[-1] <= recent[0]:
        return None
    seconds = (by_step[recent[-1]] - by_step[recent[0]]) / (recent[-1] - recent[0])
    return max(0, total-step)*seconds if seconds > 0 else None


def resume_history(train, series):
    """Include ancestor metrics only through the checkpoint actually restored."""
    merged = {tag: {p.step: p for p in points} for tag, points in series.items()}
    seen = {train.resolve()}
    restored_step = 0
    cutoff = None
    while source := read_json(train / 'run.json').get('resume_from'):
        checkpoint = Path(source)
        step = int(checkpoint.name.removeprefix('global_step_'))
        if cutoff is None:
            restored_step = step
        cutoff = min(cutoff, step) if cutoff is not None else step
        train = checkpoint.parent.parent.resolve()
        if train in seen:
            raise ValueError('Cycle in checkpoint resume history')
        seen.add(train)
        for tag, points in scalars(train / 'tensorboard').items():
            history = merged.setdefault(tag, {})
            for point in points:
                if point.step <= cutoff:
                    history.setdefault(point.step, point)
    return {tag: [points[s] for s in sorted(points)] for tag, points in merged.items()}, restored_step


def planned_updates(train):
    import yaml
    path = train / 'hydra/.hydra/config.yaml'
    if not path.exists():
        return None
    config = yaml.safe_load(path.read_text())
    manifest = read_json(train / 'data/manifest.json')
    if not manifest:
        return None
    rows = manifest['train_tasks'] * manifest['attempts_per_task_per_epoch']
    groups = rows * config['trainer']['total_epochs']
    cap = config['rollout'].get('total_rollout_steps')
    if cap is not None:
        groups = min(groups, cap)
    asynchronous = config['async_training']
    batch = (config['actor_rollout_ref']['actor']['ppo_mini_batch_size']
             * asynchronous['require_batches'] * asynchronous['trigger_parameter_sync_step'])
    return groups // batch


def concurrency(directory, submission, baseline, training):
    import yaml
    staged = submission.get('payload', {}).get('env_variables', {}).get('WORKDIR')
    baseline_limit = baseline.get('concurrency')
    if baseline_limit is None and staged:
        script = Path(staged) / 'opd/wideseek/baseline.sh'
        if script.exists():
            match = re.search(r'--concurrency\s+(\d+)', script.read_text())
            baseline_limit = int(match[1]) if match else None
    config_path = directory / 'train/hydra/.hydra/config.yaml'
    if not config_path.exists() and staged:
        config_path = Path(staged) / 'opd/wideseek/full.yaml'
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    samples = config.get('async_training', {}).get('concurrent_samples_per_replica', '?')
    workers = config.get('actor_rollout_ref', {}).get('rollout', {}).get('agent', {}).get('num_workers', '?')
    episodes = directory / ('train/episodes' if training else 'baseline')
    unfinished = sum(not p.with_name('result.json').exists() for p in episodes.glob('*/task.json'))
    return baseline_limit, samples, workers, unfinished


def show(root, run=None):
    submissions = sorted([*root.glob('*/train-submission.json'), *root.glob('*/local-run.json')], key=lambda p: p.stat().st_mtime)
    directory = (root / run) if run else (submissions[-1].parent if submissions else None)
    if directory is None:
        print('DECOMPOSER OPD  WIDESEEK  |  SETUP / WAITING FOR LAUNCH')
        setup = read_json(root.parent / 'runtime/bootstrap/status.json')
        print(setup.get('message', 'No experiment has been launched yet.'))
        print(f'Runs: {root}')
        print('Read-only | refreshes automatically; Ctrl-C exits.')
        return
    submission = read_json(directory / 'train-submission.json')
    name = submission.get('result', {}).get('job_name')
    print('Loading job status...', flush=True)
    local = read_json(directory / 'local-run.json')
    if local:
        status = {'status': local['status'], 'running_at': local['started_at'],
                  'completed_at': local.get('finished_at'), 'gpu_count': len(local['gpus']),
                  'instance_type': 'bare metal'}
        if status['status'] == 'running':
            try:
                os.kill(local['pid'], 0)
            except ProcessLookupError:
                status['status'] = 'stopped (process missing)'
        submission.setdefault('commit', local.get('commit', 'unknown'))
        if local.get('code_dir'):
            submission.setdefault('payload', {}).setdefault('env_variables', {})['WORKDIR'] = local['code_dir']
    else:
        status = job_status(name) if name else {'status': 'unknown (no submission metadata)'}
    state = status['status']
    active = state == 'running'
    now = time.time()
    end = status.get('completed_at') or now
    started = status.get('running_at')
    baseline = read_json(directory / 'baseline/manifest.json')
    train = directory / 'train'
    metadata = read_json(train / 'run.json')
    phase = ('training / validation' if metadata else 'stock baseline' if baseline else
             'model loading / preflight' if active else state)
    print(f"DECOMPOSER OPD  {directory.name}  |  {state.upper()}")
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC} | phase: {phase}")
    print(f"Job: {name or '--'} | commit: {submission.get('commit', '--')[:12]}")
    print(f"GPU allocation: {status.get('gpu_count', '--')} | {status.get('instance_type', '--')}")
    print(f"Running elapsed: {duration(end-started) if started else '--'} | "
          f"queue elapsed: {duration((started or end)-status['created_at']) if status.get('created_at') else '--'}")
    print('-' * 86)
    limit, samples, workers, unfinished = concurrency(directory, submission, baseline, bool(metadata))
    print(f"Concurrency: baseline {limit if limit is not None else '?'} | "
          f"OPD {samples} samples/replica | {workers} agent-loop workers")
    print(f"{'OPD' if metadata else 'Baseline'} unfinished episodes on disk: {unfinished} "
          '(may include interrupted attempts)')
    if baseline:
        rows = baseline.get('episodes', [])
        total = baseline['tasks'] * baseline['repetitions']
        print(f"Stock baseline: {baseline['status']} | {len(rows)}/{total} episodes")
        if rows:
            print(f"  Mean native score {sum(r['score'] for r in rows)/len(rows):.3f} | "
                  f"errors {sum(r['status'] != 'finished' for r in rows)}")
        if active and not metadata and baseline['status'] == 'running' and len(rows) >= 3:
            eta = (total-len(rows)) * (now-baseline['started_at']) / len(rows)
            print(f'  ETA baseline only: ~{duration(eta)} (observed wall throughput)')
    current_series = scalars(train / 'tensorboard')
    series, restored_step = resume_history(train, current_series)
    # ETA uses only this attempt's clock, excluding downtime before a resume.
    points = current_series.get('training/global_step', [])
    step = max(restored_step, int(points[-1].value) if points else 0)
    marker = train / 'checkpoints/latest_checkpointed_iteration.txt'
    if marker.exists():
        try:
            step = max(step, int(marker.read_text().strip()))
        except ValueError:
            pass
    total = planned_updates(train)
    print(f"Optimizer updates: {step}/{total if total is not None else '?'}")
    if restored_step:
        print(f'Resumed from step {restored_step} | history included through that checkpoint')
    if state in {'completed', 'succeeded'}:
        print('ETA: finished')
    elif not active:
        print('ETA: -- (queued, stopped, or status unavailable)')
    else:
        eta = update_eta(points, step, total)
        print(f"ETA remaining updates: ~{duration(eta)} (recent update cadence)" if eta is not None
              else 'ETA remaining updates: -- (warming up; need two logged updates)')
        print('Whole-run ETA: -- (final validation / teardown duration not yet known)')
    print('\nReward (native partial score, NOT pass rate):')
    print('  Fixed panel                       Base    Latest     Delta   Trend')
    for tag in sorted(series):
        if tag.startswith('val-core/') and re.search(r'/mean@\d+$', tag):
            values = [p.value for p in series[tag]]
            base = next((p.value for p in series[tag] if p.step == 0), None)
            first = f'{base:.3f}' if base is not None else '--'
            delta = f'{values[-1]-base:+.3f}' if base is not None else '--'
            print(f"  {tag.removeprefix('val-core/'):<32} "
                  f'{first:>6}  {values[-1]:>8.3f}  {delta:>8}   {trend(values)}')
    for tag, label in [('actor/loss', 'Actor total loss'),
                       ('actor/distillation/loss', 'Distillation loss'), ('actor/pg_loss', 'Task PG metric')]:
        values = [p.value for p in series.get(tag, [])]
        print(f"{label:20}: {trend(values)} {format(values[-1], '.4f') if values else '--'}")
    print('Sparklines: last 8 points, independently autoscaled; task PG is not an added loss here.')
    for tag in ('actor/grad_norm', 'actor/distillation/pg_clipfrac'):
        values = series.get(tag, [])
        print(f"{tag}: {values[-1].value:.4g}" if values else f'{tag}: --')
    results = [read_json(p) for p in (train / 'episodes').glob('*/result.json')]
    results = [r for r in results if r]
    print(f"OPD episodes saved: {len(results)} | errors: {sum(r.get('status') != 'finished' for r in results)}")
    demo = [read_json(p) for p in (train / 'demo-episodes').glob('*/result.json')]
    if demo:
        print(f"Demo validation episodes saved: {len(demo)} | passed: {sum(r.get('passed', False) for r in demo)}")
    files = [p for p in (directory / 'baseline.logs/eval.log', directory / 'baseline.logs/vllm.log',
                         train / 'trainer.log') if p.exists()]
    if files:
        recent = max(files, key=lambda p: p.stat().st_mtime)
        print(f'Last log write: {duration(now-recent.stat().st_mtime)} ago | {recent.name}')
    if metadata.get('error'):
        print(f"Training error: {metadata['error']}")
    print(f'Artifacts: {directory}')
    print('Read-only | Ctrl-C exits the watcher; training continues.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3] / 'decomposer-artifacts/sanity')
    parser.add_argument('--run', help='Experiment directory name (default: latest submission)')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=15)
    args = parser.parse_args()
    try:
        while True:
            if not args.once and sys.stdout.isatty():
                print('\033[2J\033[H', end='', flush=True)
            try:
                show(args.root, args.run)
            except (OSError, ValueError, KeyError, ImportError) as exc:
                print(f'OPD status unavailable: {type(exc).__name__}: {exc}')
            if args.once:
                break
            time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
