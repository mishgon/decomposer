"""Run one OPD update on four mixed-gym tasks, with two episodes each."""
import asyncio
import signal
import subprocess

from gyms.wideseek import REPO_ROOT
from opd.launch import gpu_overrides
from opd.mix.smoke_data import prepare
from opd.wideseek.local import check_inputs, reserve_gpus
from opd.wideseek.run import create_parser, main as train

# Inherit the production optimizer, LoRA, token budgets, sampling and checkpointing.
OVERRIDES = [
    'trainer.project_name=decomposer-mix-opd',
    'trainer.total_epochs=1',
    'rollout.total_rollout_steps=8',
    'trainer.val_before_train=false',
    'trainer.test_freq=-1',
]


async def main(args):
    if args.baseline or args.resume_from:
        raise ValueError('Smoke starts fresh and does not use a separate baseline')
    subprocess.run(['git', 'diff', '--exit-code', '--quiet', 'HEAD'], cwd=REPO_ROOT, check=True)
    gpu_overrides(args.gpus, args.training_gpus)
    check_inputs(args)
    with reserve_gpus(args.gpus, args.output.resolve().parent):
        overrides = [*OVERRIDES, f'trainer.experiment_name={args.output.resolve().parent.name}']
        await train(args, overrides, prepare_data=prepare, smoke=True)


if __name__ == '__main__':
    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    parser = create_parser(smoke=True)
    parser.description = __doc__
    asyncio.run(main(parser.parse_args()))
