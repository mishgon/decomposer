"""Mean training rollout reward, fixed held-out reward, and Workplace OOD reward.

These are estimates for the selected pool. Weighting does not correct selection
bias, differences in serving conditions, or uncertainty from small panels.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path

from opd.wideseek.reward import native_score, task_reward, WIDTH_REWARD


LABELS = {'train': 'Train reward', 'validation': 'Validation reward', 'ood': 'OOD reward'}


def weights(manifest, tasks, selection):
    """Weight held-out tasks to match the frozen pool dataset/pass-bucket mixture."""
    passes = {r['task_id']: r['passes'] for r in selection['evidence']}
    strata = {t['task_id']: f"{'deepdive' if t['source'].startswith('deepdive-') else t['source']}/{passes[t['task_id']]}"
              for t in tasks}
    population = Counter(strata.values())
    if population != manifest['pool_cell_counts']:
        raise ValueError('Reward strata differ from the frozen pool')
    result = {}
    for name, field in [('validation', 'heldout_task_ids')]:
        ids = manifest[field]
        counts = Counter(strata[t] for t in ids)
        if set(counts) != set(population) or len(ids) != len(set(ids)):
            raise ValueError(f'{name} panel must represent every stratum with distinct tasks')
        result[name] = {t: population[strata[t]] / sum(population.values()) / counts[strata[t]] for t in ids}
    return result


def policy_version(versions):
    value = versions.get('global_steps')
    if value is None or versions.get('min_global_steps', value) != versions.get('max_global_steps', value):
        return None
    return int(value)


class RewardSummary:
    def __init__(self, train):
        self.train = Path(train)
        path = self.train / 'data/manifest.json'
        self.manifest = json.loads(path.read_text()) if path.exists() else {}
        self.task_weights = None
        self.cache = {}
        tasks_path = self.train / 'data/tasks.jsonl'
        tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()] if tasks_path.exists() else []
        self.sources = {t['task_id']: t.get('source') for t in tasks}
        self.has_width = 'width' in self.sources.values() or self.manifest.get('source') == 'width'
        self.binary_width = self.manifest.get('width_reward') == WIDTH_REWARD
        self.gyms = {t['task_id']: 'deepdive' if t.get('source', '').startswith('deepdive-') else 'wideseek' for t in tasks}
        if self.manifest.get('stratify_by') == 'dataset_and_passes':
            selection = json.loads((self.train.parent / 'selected/selection.json').read_text())
            self.task_weights = weights(self.manifest, tasks, selection)
        elif self.has_width:
            ids = self.manifest.get('heldout_task_ids', [])
            self.task_weights = {'validation': {t: 1 / len(ids) for t in ids}}
        self.counts = {}
        if self.manifest and tasks:
            for split in ('heldout',):
                ids = self.manifest.get(f'{split}_task_ids', [])
                if ids:
                    self.counts[split] = Counter(self.gyms[t] for t in ids)

    def panels(self):
        """Read immutable completed trajectories once; retain the latest retry."""
        expected = set().union(*(set(w) for w in self.task_weights.values()))
        panels = defaultdict(dict)
        for path in self.train.joinpath('episodes').glob('*/trajectory.json'):
            stamp = path.stat().st_mtime_ns
            cached = self.cache.get(path)
            if cached is None or cached[0] != stamp:
                # Select held-out evaluations, including historical runs with probes.
                try:
                    d = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue  # The trajectory is still being written.
                source, tid = d.get('data_source', ''), d.get('task_id')
                v = policy_version(d.get('weight_versions', {}))
                record = None
                if tid in expected and v is not None and source.endswith('/heldout') and d.get('result'):
                    record = (source.rsplit('/', 1)[-1], v, tid, d['result'])
                self.cache[path] = (stamp, record)
            record = self.cache[path][1]
            if record:
                split, v, tid, result = record
                previous = panels[(split, v)].get(tid)
                if previous is None or result['finished_at'] > previous['finished_at']:
                    panels[(split, v)][tid] = result
        return panels

    def metrics(self, logged, step):
        """Only aggregate a validation after every constituent gym was logged."""
        output = {}
        # An old mean F1 cannot be converted to a binary success rate.
        # Exact historical optimizer batches are not identified by policy versions.
        if 'critic/rewards/mean' in logged and (not self.has_width or self.binary_width):
            output['rewards/train'] = logged['critic/rewards/mean']
        has_validation = any('/heldout/reward/mean@' in tag for tag in logged)
        panels = self.panels() if self.task_weights and has_validation else {}
        for name, split in [('validation', 'heldout')]:
            tags = {tag: value for tag, value in logged.items()
                    if tag.startswith('val-core/') and f'/{split}/reward/mean@' in tag
                    and not tag.startswith('val-core/mix/')}
            counts = self.counts.get(split)
            if not tags:
                continue
            if self.task_weights:
                expected = self.task_weights[name]
                results = panels.get((split, step), {})
                if not expected or not set(expected) <= results.keys():
                    continue
                # Verify task scores reproduce the trainer's recorded reward convention.
                # This detects mismatched retries, versions, missing tasks and judge failures.
                scores, rewards = {}, {}
                for tid in expected:
                    result = results[tid]
                    rewards[tid] = task_reward(result, self.sources[tid])
                    scores[tid] = rewards[tid] if self.binary_width else native_score(result)
                for gym, count in counts.items():
                    tag = f'val-core/{gym}/{split}/reward/mean@1'
                    if tag not in tags:
                        break
                    values = [v for tid, v in scores.items() if self.gyms[tid] == gym]
                    if len(values) != count or not math.isclose(sum(values) / count, tags[tag], abs_tol=1e-6):
                        raise ValueError(f'Raw results disagree with logged validation: {tag}, step {step}')
                else:
                    output[f'rewards/{name}'] = sum(rewards[t] * w for t, w in expected.items())
                continue
            # Older single-gym and mixed runs retain their fixed-panel task means.
            if counts:
                keys = {gym: f'val-core/{gym}/{split}/reward/mean@1' for gym in counts}
                if all(tag in tags for tag in keys.values()):
                    output[f'rewards/{name}'] = sum(tags[keys[gym]] * n for gym, n in counts.items()) / sum(counts.values())
            elif len(tags) == 1:
                output[f'rewards/{name}'] = next(iter(tags.values()))
        ood = {k: v for k, v in logged.items() if k.startswith('val-core/workplace-assistant/validation20/reward/mean@')}
        if ood:
            output['rewards/ood'] = ood[max(ood, key=lambda k: int(k.rsplit('@', 1)[1]))]
        return output


@lru_cache(maxsize=8)
def reader(train, manifest_stamp):
    return RewardSummary(train)


def summarize(train, logged, step):
    train = Path(train).resolve()
    manifest = train / 'data/manifest.json'
    stamp = manifest.stat().st_mtime_ns if manifest.exists() else None
    return reader(str(train), stamp).metrics(logged, step)


def series(train, logged_series):
    """Return three curves from normalized {step, value, time} scalar records."""
    by_step = defaultdict(dict)
    times = defaultdict(float)
    for tag, points in logged_series.items():
        if not (tag.startswith('val-core/') or tag == 'critic/rewards/mean'):
            continue
        for p in points:
            by_step[p['step']][tag] = p['value']
            times[p['step']] = max(times[p['step']], p['time'])
    output = {f'rewards/{key}': [] for key in LABELS}
    for step, values in sorted(by_step.items()):
        for tag, value in summarize(train, values, step).items():
            output[tag].append({'step': step, 'value': value, 'time': times[step]})
    return output


def main():
    """Backfill/follow an already running trainer without restarting training."""
    import argparse
    import fcntl
    import time
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from opd.wideseek.watch import scalars

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--clearml-task-id')
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    train = args.run / 'train'
    state_path = train / 'reward-summary.json'
    with (train / 'reward-summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        published = json.loads(state_path.read_text()) if state_path.exists() else {}
        logger = None
        if args.clearml_task_id:
            from clearml import Task
            logger = Task.get_task(task_id=args.clearml_task_id).get_logger()
        writer = EventFileWriter(str(train / 'tensorboard/reward-summary'))
        try:
            while True:
                logged = {tag: [{'step': p.step, 'value': p.value, 'time': p.wall_time} for p in points]
                          for tag, points in scalars(train / 'tensorboard').items()}
                curves = series(train, logged)
                for tag, points in curves.items():
                    for p in points:
                        key = f"{tag}/{p['step']}"
                        if published.get(key) == p['value']:
                            continue
                        if logger:
                            title, name = tag.split('/', 1)
                            logger.report_scalar(title, name, value=p['value'], iteration=p['step'])
                        writer.add_event(Event(step=p['step'], wall_time=p['time'],
                            summary=Summary(value=[Summary.Value(tag=tag, simple_value=p['value'])])))
                        published[key] = p['value']
                writer.flush()
                if logger:
                    logger.flush(wait=True)
                temporary = state_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(published, indent=2) + '\n')
                temporary.replace(state_path)
                print(json.dumps({tag: points[-1] if points else None for tag, points in curves.items()}), flush=True)
                status = json.loads((train / 'run.json').read_text())['status']
                if not args.follow or status in {'completed', 'failed', 'stopped'}:
                    break
                time.sleep(30)
        finally:
            writer.close()


if __name__ == '__main__':
    main()
