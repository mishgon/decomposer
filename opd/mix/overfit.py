"""Train on a frozen selection with the production OPD recipe."""
import asyncio
from collections import Counter
from functools import partial
import hashlib
import json
import random
import signal
import subprocess

from gyms.wideseek import REPO_ROOT
from gyms.wideseek.prepare import agent_input
from opd.launch import gpu_overrides
from opd.wideseek.local import check_inputs, reserve_gpus
from opd.wideseek.prepare import DEPTH_VALIDATION_PANEL, VALIDATION_PANEL
from opd.wideseek.run import create_parser, main as train
from opd.workplace_assistant.prepare import validation_rows

OVERRIDES = ['trainer.project_name=decomposer-mix-opd', 'trainer.total_epochs=1']


def prepare(source, output, *, steps=32):
    import pandas as pd

    raw = source.read_bytes()
    tasks = [json.loads(line) for line in raw.splitlines() if line.strip()]
    heldout = {t['question_sha256'] for panel in (VALIDATION_PANEL, DEPTH_VALIDATION_PANEL)
               for t in json.loads(panel.read_text())['heldout']}
    if (not tasks or len({t['task_id'] for t in tasks}) != len(tasks)
            or len({t['question_sha256'] for t in tasks}) != len(tasks)):
        raise ValueError('Training requires distinct selected tasks')
    if steps < 1 or len(tasks) > steps * 8:
        raise ValueError('The update budget must visit every selected task at least once')
    if any(t['source'] not in {'depth', 'width', 'deepdive-qa_rl', 'deepdive-qa_sft'} for t in tasks):
        raise ValueError('Unsupported training source')
    if any(t['question_sha256'] in heldout for t in tasks):
        raise ValueError('Selected training tasks overlap the frozen holdout')

    def row(task, split):
        gym = 'deepdive' if task['source'].startswith('deepdive-') else 'wideseek'
        return {'data_source': f'{gym}/{split}', 'agent_name': 'wideseek_decomposer',
                'prompt': agent_input(task)['messages'],
                'reward_model': {'style': 'rule', 'ground_truth': ''},
                'extra_info': {'task_id': task['task_id']}}

    # Balance exposure, including the final partial pass through the pool.
    rng = random.Random(42)
    ordered = list(tasks)
    rng.shuffle(ordered)
    training = [row(ordered[i % len(tasks)], 'train') for i in range(steps * 8)]
    rng.shuffle(training)
    probes = ordered[:6]
    workplace = validation_rows()
    validation = [row(t, 'train_probe') for t in probes] + workplace
    output.mkdir(parents=True, exist_ok=False)
    (output / 'tasks.jsonl').write_bytes(raw)
    pd.DataFrame(training).to_parquet(output / 'train.parquet', index=False)
    pd.DataFrame(validation).to_parquet(output / 'evaluation.parquet', index=False)
    manifest = {'overfit': True, 'seed': 42, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'task_ids': [t['task_id'] for t in tasks],
                'train_task_ids': [t['task_id'] for t in tasks], 'train_tasks': len(tasks),
                'heldout_task_ids': [], 'heldout_tasks': 0, 'train_probe_tasks': len(probes),
                'train_probe_task_ids': [t['task_id'] for t in probes],
                'training_sources': dict(Counter(t['source'] for t in tasks)),
                'workplace_validation_tasks': len(workplace), 'training_episodes': steps * 8,
                'optimizer_steps': steps,
                'evaluation': 'Up to six fixed training probes and frozen Workplace-20; pass@1',
                'files': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in ('tasks.jsonl', 'train.parquet', 'evaluation.parquet')}}
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
    from opd.workplace_assistant.preflight import check
    await check()
    with reserve_gpus(args.gpus, args.output.resolve().parent):
        await train(args, [*OVERRIDES, f'rollout.total_rollout_steps={args.steps * 8}',
                          f'trainer.experiment_name={args.output.resolve().parent.name}'],
                    prepare_data=partial(prepare, steps=args.steps), smoke=True)


if __name__ == '__main__':
    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    parser = create_parser(smoke=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.description = __doc__
    asyncio.run(main(parser.parse_args()))
