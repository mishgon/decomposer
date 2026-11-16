"""Report in-band progress to ClearML, embedded or against an existing run."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import fcntl
from itertools import groupby
import json
import math
import os
from pathlib import Path
import time
from threading import Event, Thread
from types import SimpleNamespace
import warnings

import httpx

from evals.wideseek.band import qualifies_band, resolved
from gyms.wideseek.runtime import save


def domain(task):
    for prefix, name in (('depth-', 'WideSeek depth'), ('width-', 'WideSeek width'),
                         ('deepdive-qa_rl-', 'DeepDive QA RL'),
                         ('deepdive-qa_sft-', 'DeepDive QA SFT')):
        if task.startswith(prefix):
            return name
    return 'Other'


class Totals:
    def __init__(self, repetitions, threshold, max_pass_rate):
        self.repetitions, self.threshold, self.max_pass_rate = repetitions, threshold, max_pass_rate
        self.counts, self.attempts, self.passes = Counter(), Counter(), Counter()
        self.qualified, self.histogram = Counter(), Counter()

    def add(self, row):
        self.counts['completed'] += 1
        outcome = resolved(row, self.threshold)
        if outcome is None:
            self.counts['unresolved'] += 1
            return
        self.counts['resolved'] += 1
        self.counts['passed'] += outcome
        task = row['task_id']
        if self.attempts[task] >= self.repetitions:
            self.histogram[self.passes[task]] -= 1
            self.qualified[domain(task)] -= qualifies_band(
                self.passes[task], self.attempts[task], self.max_pass_rate)
        self.attempts[task] += 1
        self.passes[task] += outcome
        if self.attempts[task] >= self.repetitions:
            self.histogram[self.passes[task]] += 1
            self.qualified[domain(task)] += qualifies_band(
                self.passes[task], self.attempts[task], self.max_pass_rate)
        evaluation = row.get('evaluation', {})
        score = evaluation.get('score')
        if (evaluation.get('status') == 'scored' and type(score) in (int, float)
                and math.isfinite(score) and 0 <= score <= 1):
            self.counts['native_sum'] += score
            self.counts['native_count'] += 1
        self.counts['duration_sum'] += max(0, row['finished_at'] - row['started_at']) / 60

    def metrics(self, total, domains):
        values = {
            ('Episodes', 'Completed'): self.counts['completed'],
            ('Episodes', 'Resolved'): self.counts['resolved'],
            ('Episodes', 'Unresolved'): self.counts['unresolved'],
            ('Episodes', 'Remaining'): total - self.counts['completed'],
            ('In-band tasks', 'All'): sum(self.qualified.values()),
            ('Evaluated tasks', 'All attempts resolved'): sum(self.histogram.values()),
        }
        values.update({('In-band tasks', name): self.qualified[name] for name in domains})
        values.update({('Successes per evaluated task', f'{i}/{self.repetitions}'): self.histogram[i]
                       for i in range(self.repetitions + 1)})
        if self.counts['resolved']:
            values['Pass rate (%)', 'Resolved episodes'] = 100 * self.counts['passed'] / self.counts['resolved']
            values['Episode duration (minutes)', 'Mean resolved'] = self.counts['duration_sum'] / self.counts['resolved']
        if self.counts['native_count']:
            values['Native score', 'Mean scored episodes'] = self.counts['native_sum'] / self.counts['native_count']
        return values


def timeline(rows, manifest, threshold, max_pass_rate):
    """Cumulative minute buckets; imported results precede this run at minute zero."""
    settings = manifest['settings']
    total = len(settings['tasks']) * settings['repetitions']
    domains = sorted({domain(task) for task in settings['tasks']})
    totals = Totals(settings['repetitions'], threshold, max_pass_rate)
    points, fresh = {}, 0
    started = manifest['started_at']
    def minute(row):
        return max(0, int((row['finished_at'] - started) / 60))
    for step, group in groupby(sorted(rows, key=lambda r: r['finished_at']), minute):
        for row in group:
            totals.add(row)
            fresh += row['finished_at'] >= started
        values = totals.metrics(total, domains)
        if step:
            values['Throughput (episodes/hour)', 'Since collection start'] = fresh * 60 / (step + 1)
        points[step] = values
    return points, totals.metrics(total, domains)


def snapshot(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    mode, = manifest['settings']['modes']
    rows, unfinished = [], 0
    for attempt in (run / mode).glob('*/attempt-*'):
        try:
            row = json.loads((attempt / 'result.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            unfinished += 1
            continue
        row['trace_available'] = (attempt / row['execution_directory'] / 'trace.json').is_file()
        rows.append(row)
    return manifest, rows, unfinished


def collector_gpu_metrics(api, job_id):
    """Read only the GPUs assigned to the collector, not the reporter's host."""
    with httpx.Client(base_url=api.rstrip('/') + '/', timeout=10,
                      headers={'Authorization': f'Bearer {os.environ["OCC_TOKEN"]}'}) as client:
        response = client.get(f'jobs/{job_id}/attempts')
        response.raise_for_status()
        attempts = [a for a in response.json()['items'] if a['status'] == 'Running']
        if not attempts:
            return {}
        values = {}
        for worker in max(attempts, key=lambda a: a['attemptNumber'])['workers']:
            response = client.get(f'nodes/{worker["nodeId"]}')
            response.raise_for_status()
            inventory = response.json()['inventory']
            # A stale observation is missing telemetry, not zero utilization.
            observed = datetime.fromisoformat(inventory['observedAt'].replace('Z', '+00:00')).timestamp()
            if time.time() - observed > 180:
                continue
            for gpu in inventory['gpus']:
                if gpu['id'] not in worker['gpuDeviceIds']:
                    continue
                series = f'Worker {worker["workerIndex"]} GPU {gpu["index"]}'
                for title, field, divisor in (
                        ('GPU utilization (%)', 'utilizationPercent', 1),
                        ('GPU memory (GiB)', 'memoryUsedBytes', 1024 ** 3)):
                    if gpu.get(field) is not None:
                        values[title, series] = gpu[field] / divisor
        return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Directory containing manifest.json')
    parser.add_argument('--state', type=Path, required=True, help='Persistent reporter state')
    parser.add_argument('--project', default='decomposer-inband')
    parser.add_argument('--name')
    parser.add_argument('--success-threshold', type=float, default=.9)
    parser.add_argument('--max-pass-rate', type=float, default=.5)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--collector-job', help='OCCtopus job ID for a separate CPU reporter')
    parser.add_argument('--occtopus-api', help='OCCtopus API v1 URL; authenticates with OCC_TOKEN')
    args = parser.parse_args()
    if args.interval <= 0 or not 0 <= args.max_pass_rate <= 1 or not 0 <= args.success_threshold <= 1:
        parser.error('Use a positive interval and thresholds in [0, 1]')
    if bool(args.collector_job) != bool(args.occtopus_api):
        parser.error('--collector-job and --occtopus-api must be supplied together')
    report(args)


def report(args, stop=None):
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(args.state.read_text()) if args.state.exists() else {}
        identity = {'run': str(args.run.resolve()), 'threshold': args.success_threshold,
                    'max_pass_rate': args.max_pass_rate}
        if state and state['identity'] != identity:
            raise ValueError('Reporter state belongs to another run or selection rule')
        from clearml import Task
        task = Task.init(project_name=args.project, task_name=args.name or args.run.parent.name,
            task_type=Task.TaskTypes.testing, reuse_last_task_id=state.get('task_id', False),
            continue_last_task=0 if state.get('task_id') else False,
            auto_connect_frameworks=False, auto_connect_arg_parser=False,
            auto_resource_monitoring=not bool(getattr(args, 'collector_job', None)),
            auto_connect_streams=False, output_uri=False)
        state.update(identity=identity, task_id=task.id, url=task.get_output_log_web_page())
        save(args.state, state)
        manifest = json.loads((args.run / 'manifest.json').read_text())
        settings = manifest['settings']
        launch = json.loads((args.run.parent / 'launch.json').read_text())
        task.connect({**identity, 'collector_commit': manifest['revision'], 'x_axis': 'elapsed minutes',
            'tasks': len(settings['tasks']), 'repetitions': settings['repetitions'],
            'concurrency': settings['concurrency'], 'episode_timeout': settings['episode_timeout'],
            'models': settings['model_profiles'], 'checkpoint': launch.get('initialization')}, name='Collection')
        logger = task.get_logger()
        print(f'ClearML: {state["url"]}', flush=True)
        while True:
            manifest, rows, unfinished = snapshot(args.run)
            points, current = timeline(rows, manifest, args.success_threshold, args.max_pass_rate)
            elapsed = max(1, time.time() - manifest['started_at'])
            minute = max(0, int(elapsed / 60))
            current['Throughput (episodes/hour)', 'Since collection start'] = (
                sum(row['finished_at'] >= manifest['started_at'] for row in rows) * 3600 / elapsed)
            points[minute] = current
            reported = state.setdefault('reported', {})
            for step, values in sorted(points.items()):
                encoded = json.dumps(sorted((title, series, value) for (title, series), value in values.items()))
                if reported.get(str(step)) == encoded:
                    continue
                for (title, series), value in values.items():
                    logger.report_scalar(title, series, value, iteration=step)
                reported[str(step)] = encoded
            launch = json.loads((args.run.parent / 'launch.json').read_text())
            status = launch['status']
            logger.report_scalar('Episodes', 'Active', unfinished if status == 'running' else 0, iteration=minute)
            outage = args.run / 'outage.json'
            paused = outage.exists() and json.loads(outage.read_text()).get('status') == 'paused'
            logger.report_scalar('Service outage', 'Paused', int(paused), iteration=minute)
            if getattr(args, 'collector_job', None):
                try:
                    for (title, series), value in collector_gpu_metrics(args.occtopus_api, args.collector_job).items():
                        logger.report_scalar(title, series, value, iteration=minute)
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    warnings.warn(f'GPU telemetry unavailable ({type(exc).__name__}); episode reporting continues')
            task.flush(wait_for_uploads=True)
            state.update(updated_at=time.time(), completed=len(rows), active=unfinished,
                         qualifying=current['In-band tasks', 'All'], collection_status=status)
            save(args.state, state)
            print(f'Reported {len(rows)} episodes, {state["qualifying"]} qualifying tasks; collection {status}', flush=True)
            if status in {'completed', 'failed', 'interrupted', 'stopped'} or (stop and stop.is_set()):
                if status != 'completed':
                    task.mark_stopped(status_message=f'Collection {status}')
                task.close()
                return
            if stop is None:
                time.sleep(args.interval)
            else:
                stop.wait(args.interval)


@contextmanager
def collection_reporter(args):
    """Follow the collector within its job; reporting must not cancel episodes."""
    stop = Event()
    run = args.output.resolve()
    options = SimpleNamespace(run=run, state=run.parent / 'clearml.json',
        project=args.clearml_project, name=args.clearml_name,
        success_threshold=args.success_threshold, max_pass_rate=args.max_pass_rate, interval=60)

    def follow():
        try:
            while not (run / 'manifest.json').is_file():
                if stop.wait(1):
                    return
            report(options, stop)
        except Exception as exc:
            warnings.warn(f'ClearML reporting stopped ({type(exc).__name__}: {exc}); collection continues')

    thread = Thread(target=follow, name='inband-clearml', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=30)
        if thread.is_alive():
            warnings.warn('ClearML flush exceeded 30 seconds; persistent results remain available for backfill')


if __name__ == '__main__':
    main()
