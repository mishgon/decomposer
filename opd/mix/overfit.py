"""Overfit six selected tasks for 32 updates with the production OPD recipe."""
import asyncio
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

OVERRIDES = ['trainer.project_name=decomposer-mix-opd', 'trainer.total_epochs=1',
             'rollout.total_rollout_steps=256']


def prepare(source, output):
    import pandas as pd

    raw = source.read_bytes()
    tasks = [json.loads(line) for line in raw.splitlines() if line.strip()]
    heldout = {t['question_sha256'] for panel in (VALIDATION_PANEL, DEPTH_VALIDATION_PANEL)
               for t in json.loads(panel.read_text())['heldout']}
    if (len(tasks) != 6 or len({t['task_id'] for t in tasks}) != 6
            or len({t['question_sha256'] for t in tasks}) != 6):
        raise ValueError('Overfit requires exactly six distinct selected tasks')
    if any(t['question_sha256'] in heldout for t in tasks):
        raise ValueError('Selected training tasks overlap the frozen holdout')

    def row(task, split):
        gym = 'deepdive' if task['source'].startswith('deepdive-') else 'wideseek'
        return {'data_source': f'{gym}/{split}', 'agent_name': 'wideseek_decomposer',
                'prompt': agent_input(task)['messages'],
                'reward_model': {'style': 'rule', 'ground_truth': ''},
                'extra_info': {'task_id': task['task_id']}}

    # 256 episodes / production batch 8 = 32 updates; balance exposure across tasks.
    training = [row(tasks[i % 6], 'train') for i in range(256)]
    random.Random(42).shuffle(training)
    workplace = validation_rows()
    validation = [row(t, 'train_probe') for t in tasks] + workplace
    output.mkdir(parents=True, exist_ok=False)
    (output / 'tasks.jsonl').write_bytes(raw)
    pd.DataFrame(training).to_parquet(output / 'train.parquet', index=False)
    pd.DataFrame(validation).to_parquet(output / 'evaluation.parquet', index=False)
    manifest = {'overfit': True, 'seed': 42, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'task_ids': [t['task_id'] for t in tasks],
                'train_task_ids': [t['task_id'] for t in tasks], 'train_tasks': 6,
                'heldout_task_ids': [], 'heldout_tasks': 0, 'train_probe_tasks': 6,
                'workplace_validation_tasks': len(workplace), 'training_episodes': 256,
                'evaluation': 'All six training tasks and frozen Workplace-20; pass@1',
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
    from opd.workplace_assistant.preflight import check
    await check()
    with reserve_gpus(args.gpus, args.output.resolve().parent):
        await train(args, [*OVERRIDES, f'trainer.experiment_name={args.output.resolve().parent.name}'],
                    prepare_data=prepare, smoke=True)


if __name__ == '__main__':
    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    parser = create_parser(smoke=True)
    parser.description = __doc__
    asyncio.run(main(parser.parse_args()))
