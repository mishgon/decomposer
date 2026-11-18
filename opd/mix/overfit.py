"""Train on a frozen selection with the production OPD recipe."""
import asyncio
from collections import Counter
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import shutil
import signal
import subprocess

from gyms.wideseek import REPO_ROOT
from gyms.wideseek.prepare import agent_input
from opd.launch import gpu_overrides
from opd.mix.run import OVERRIDES as MIX_OVERRIDES
from opd.wideseek.local import check_inputs, reserve_gpus
from opd.wideseek.prepare import DEPTH_VALIDATION_PANEL, VALIDATION_PANEL, check
from opd.wideseek.run import create_parser, main as train
from opd.workplace_assistant.prepare import validation_rows
from opd.wideseek.reward import WIDTH_REWARD

OVERRIDES = [*MIX_OVERRIDES, 'trainer.total_epochs=1']


def frozen_manifest(source, directory, steps, holdout_tasks):
    """Validate an already selected split without drawing a new holdout."""
    import pandas as pd

    manifest = check(directory)
    if (hashlib.sha256(source.read_bytes()).hexdigest() != manifest['source_sha256']
            or source.read_bytes() != (directory / 'tasks.jsonl').read_bytes()):
        raise ValueError('Frozen data and selected task pool differ')
    train = pd.read_parquet(directory / 'train.parquet')
    evaluation = pd.read_parquet(directory / 'evaluation.parquet')
    train_ids = {row['task_id'] for row in train.extra_info}
    heldout_ids = {row['task_id'] for row in
                   evaluation[evaluation.data_source.str.endswith('/heldout')].extra_info}
    if (len(train) != steps * 8 or manifest['optimizer_steps'] != steps
            or len(heldout_ids) != holdout_tasks
            or train_ids != set(manifest['train_task_ids'])
            or heldout_ids != set(manifest['heldout_task_ids'])
            or train_ids & heldout_ids):
        raise ValueError('Frozen training budget or holdout differs')
    return manifest


def stage_frozen(source, output, *, directory, steps, holdout_tasks):
    import pandas as pd

    manifest = frozen_manifest(source, directory, steps, holdout_tasks)
    shutil.copytree(directory, output)
    path = output / 'evaluation.parquet'
    evaluation = pd.read_parquet(path)
    probes = evaluation.data_source.str.endswith('/train_probe')
    if probes.any():
        evaluation.loc[~probes].to_parquet(path, index=False)
        manifest.update(train_probe_tasks=0, train_probe_task_ids=[],
                        evaluation='Fixed held-out and Workplace-20; pass@1')
        manifest['files']['evaluation.parquet'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['width_reward'] = WIDTH_REWARD
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def prepare(source, output, *, steps=32, holdout_tasks=0, prioritize_strict=False):
    import pandas as pd

    raw = source.read_bytes()
    tasks = [json.loads(line) for line in raw.splitlines() if line.strip()]
    heldout = {t['question_sha256'] for panel in (VALIDATION_PANEL, DEPTH_VALIDATION_PANEL)
               for t in json.loads(panel.read_text())['heldout']}
    if (not tasks or len({t['task_id'] for t in tasks}) != len(tasks)
            or len({t['question_sha256'] for t in tasks}) != len(tasks)):
        raise ValueError('Training requires distinct selected tasks')
    if any(t['source'] not in {'depth', 'width', 'deepdive-qa_rl', 'deepdive-qa_sft'} for t in tasks):
        raise ValueError('Unsupported training source')
    if any(t['question_sha256'] in heldout for t in tasks):
        raise ValueError('Selected training tasks overlap the frozen holdout')

    rng = random.Random(42)
    passes = {}
    if prioritize_strict:
        selection = json.loads(source.with_name('selection.json').read_text())
        if (selection['tasks_sha256'] != hashlib.sha256(raw).hexdigest()
                or selection['repetitions'] != 4):
            raise ValueError('Pass-bucket selection requires the unchanged four-attempt snapshot')
        for entry in selection['evidence']:
            count = entry['passes']
            if (count not in (0, 1, 2) or len(entry['attempts']) != 4
                    or sum(a['passed'] for a in entry['attempts']) != count
                    or passes.get(entry['task_id'], count) != count):
                raise ValueError('Invalid or conflicting pass-bucket evidence')
            passes[entry['task_id']] = count
        if not {t['task_id'] for t in tasks} <= passes.keys():
            raise ValueError('Missing pass-bucket evidence')
        strict = sorted((t for t in tasks if passes[t['task_id']] > 0), key=lambda t: t['task_id'])
        zeros = sorted((t for t in tasks if passes[t['task_id']] == 0), key=lambda t: t['task_id'])
        size = steps * 8 + holdout_tasks
        if not len(strict) <= size <= len(tasks):
            raise ValueError('Pool must fit every strict task and fill the training budget')
        tasks = strict + rng.sample(zeros, size - len(strict))

    def row(task, split):
        gym = 'deepdive' if task['source'].startswith('deepdive-') else 'wideseek'
        return {'data_source': f'{gym}/{split}', 'agent_name': 'wideseek_decomposer',
                'prompt': agent_input(task)['messages'],
                'reward_model': {'style': 'rule', 'ground_truth': ''},
                'extra_info': {'task_id': task['task_id']}}

    # Balance exposure, including the final partial pass through the pool.
    ordered = list(tasks)
    holdout = []
    strata = {t['task_id']: passes[t['task_id']] if prioritize_strict else t['source'] for t in tasks}
    if holdout_tasks:
        groups = {group: sorted((t for t in tasks if strata[t['task_id']] == group), key=lambda t: t['task_id'])
                  for group in sorted(set(strata.values()))}
        if (not len(groups) <= holdout_tasks <= len(tasks) - len(groups)
                or any(len(group) < 2 for group in groups.values())):
            raise ValueError('Holdout must leave training and validation tasks in every source or pass bucket')
        counts = dict.fromkeys(groups, 1)
        for _ in range(holdout_tasks - len(groups)):
            source = max((s for s in groups if counts[s] < len(groups[s]) - 1),
                         key=lambda s: len(groups[s]) * holdout_tasks / len(tasks) - counts[s])
            counts[source] += 1
        ordered = []
        for source, group in groups.items():
            rng.shuffle(group)
            holdout.extend(group[:counts[source]])
            ordered.extend(group[counts[source]:])
    if steps < 1 or len(ordered) > steps * 8:
        raise ValueError('The update budget must visit every selected training task at least once')
    rng.shuffle(ordered)
    training = [row(ordered[i % len(ordered)], 'train') for i in range(steps * 8)]
    rng.shuffle(training)
    workplace = validation_rows()
    validation = [row(t, 'heldout') for t in holdout] + workplace
    output.mkdir(parents=True, exist_ok=False)
    (output / 'tasks.jsonl').write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in tasks))
    pd.DataFrame(training).to_parquet(output / 'train.parquet', index=False)
    pd.DataFrame(validation).to_parquet(output / 'evaluation.parquet', index=False)
    manifest = {'overfit': not bool(holdout), 'width_reward': WIDTH_REWARD,
                'seed': 42, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'task_ids': [t['task_id'] for t in tasks],
                'train_task_ids': [t['task_id'] for t in ordered], 'train_tasks': len(ordered),
                'heldout_task_ids': [t['task_id'] for t in holdout], 'heldout_tasks': len(holdout),
                'heldout_sources': dict(Counter(t['source'] for t in holdout)), 'train_probe_tasks': 0,
                'train_probe_task_ids': [],
                'training_sources': dict(Counter(t['source'] for t in ordered)),
                'workplace_validation_tasks': len(workplace), 'training_episodes': steps * 8,
                'optimizer_steps': steps,
                'evaluation': f'{len(holdout)} disjoint heldout tasks '
                              'and frozen Workplace-20; pass@1',
                'files': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in ('tasks.jsonl', 'train.parquet', 'evaluation.parquet')}}
    if prioritize_strict:
        manifest.update(stratify_by='passes', repetitions=4, prioritize_strict=True,
                        pool_pass_counts=dict(Counter(passes[t['task_id']] for t in tasks)),
                        train_pass_counts=dict(Counter(passes[t['task_id']] for t in ordered)),
                        heldout_pass_counts=dict(Counter(passes[t['task_id']] for t in holdout)))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


async def main(args):
    if args.baseline or args.resume_from:
        raise ValueError('Overfit starts fresh; veRL evaluates the initial policy at step zero')
    subprocess.run(['git', 'diff', '--exit-code', '--quiet', 'HEAD'], cwd=REPO_ROOT, check=True)
    gpu_overrides(args.gpus, args.training_gpus)
    check_inputs(args)
    selection = args.source.with_name('selection.json')
    if selection.exists():
        selected = json.loads(selection.read_text())
        initialization = json.loads((args.model / 'opd_initialization.json').read_text())
        if selected['initialization'] != initialization:
            raise ValueError('Training checkpoint differs from in-band selection checkpoint')
        if selected['tasks_sha256'] != hashlib.sha256(args.source.read_bytes()).hexdigest():
            raise ValueError('Frozen in-band tasks changed')
    prepare_data = partial(prepare, steps=args.steps, holdout_tasks=args.holdout_tasks,
                           prioritize_strict=args.prioritize_strict)
    overrides = list(OVERRIDES)
    if args.prepared_data:
        if args.prioritize_strict:
            raise ValueError('Cannot resample an already prepared split')
        manifest = frozen_manifest(args.source, args.prepared_data, args.steps, args.holdout_tasks)
        initialization = json.loads((args.model / 'opd_initialization.json').read_text())
        if manifest['initialization'] != initialization:
            raise ValueError('Prepared split and training checkpoint differ')
        overrides.append(f"wideseek.timeout={manifest['episode_timeout']}")
        prepare_data = partial(stage_frozen, directory=args.prepared_data,
                               steps=args.steps, holdout_tasks=args.holdout_tasks)
    from opd.workplace_assistant.preflight import check
    await check()
    with reserve_gpus(args.gpus, args.output.resolve().parent):
        await train(args, [*overrides, f'rollout.total_rollout_steps={args.steps * 8}',
                          f'trainer.experiment_name={args.output.resolve().parent.name}'],
                    prepare_data=prepare_data, smoke=True)


if __name__ == '__main__':
    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    parser = create_parser(smoke=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--prepared-data', type=Path,
                        help='Reuse a checksummed prepared split, episode order and time budget')
    parser.add_argument('--holdout-tasks', type=int, required=True,
                        help='Fixed seed-42 holdout stratified by source; use 0 only for deliberate overfit runs')
    parser.add_argument('--prioritize-strict', action='store_true',
                        help='Keep every 1/4 and 2/4 task, fill the update budget with 0/4 tasks, '
                             'and stratify holdout by pass bucket; requires selection.json')
    parser.description = __doc__
    asyncio.run(main(parser.parse_args()))
