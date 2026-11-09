"""Collect with dedicated workers, restarting them after unsafe cancellation."""
import asyncio
import json
import random
import time
from pathlib import Path

from decomposer.agent_server import agent_server
from gyms.wideseek import REPO_ROOT
from gyms.wideseek.run import cli
from gyms.wideseek.runtime import save
from sft.wideseek.outage import WorkerCleanupError, recoverable_cleanup
from sft.wideseek.run import create_collection_parser, main as collect


async def run_with_workers(args, config, collection):
    resets = 0
    while True:
        try:
            async with agent_server(config, port=args.worker_port,
                                    n_jobs_per_worker=args.worker_slots, startup_timeout=180,
                                    python_executable=args.worker_python) as url:
                args.worker_url = url
                if resets:
                    save(args.output / 'worker-recovery.json',
                         dict(status='running', resets=resets, recovered_at=time.time()))
                await collection(args)
            return
        except WorkerCleanupError as error:
            # agent_server has terminated and reaped its process group at this point.
            # Preserve unsafe attempts but remove them from completed-task indexing.
            for path in args.output.glob('*/*/attempt-*/result.json'):
                result = json.loads(path.read_text())
                if recoverable_cleanup(result):
                    path.replace(path.parent / result['execution_directory'] / 'cleanup-failed-result.json')
            ceiling = min(300, 5 * 2 ** min(resets, 6))
            delay = random.uniform(ceiling / 2, ceiling)
            resets += 1
            save(args.output / 'worker-recovery.json', dict(status='restarting', resets=resets,
                 reason=str(error), next_retry_at=time.time() + delay))
            print(f'Worker server stopped; resuming unfinished tasks in {delay:.1f}s', flush=True)
            args.resume = True
            await asyncio.sleep(delay)


async def main(args):
    await run_with_workers(args, REPO_ROOT / 'gyms/wideseek/langgraph.json', collect)


def create_parser():
    parser = create_collection_parser()
    parser.add_argument('--worker-port', type=int, default=18084)
    parser.add_argument('--worker-slots', type=int, default=512)
    parser.add_argument('--worker-python', type=Path)
    return parser


if __name__ == '__main__':
    cli(main, parser=create_parser())
