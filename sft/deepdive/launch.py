"""Launch DeepDive collection with dedicated local researcher workers."""
from pathlib import Path

from decomposer.agent_server import agent_server
from gyms.deepdive import REPO_ROOT
from gyms.wideseek.run import cli
from sft.deepdive.run import create_collection_parser, main as collect


async def main(args):
    async with agent_server(REPO_ROOT / "gyms/deepdive/langgraph.json",
                            port=args.worker_port, n_jobs_per_worker=args.worker_slots,
                            startup_timeout=180, python_executable=args.worker_python) as url:
        args.worker_url = url
        await collect(args)


def create_parser():
    parser = create_collection_parser()
    parser.description = __doc__
    parser.add_argument("--worker-port", type=int, default=18085)
    parser.add_argument("--worker-slots", type=int, default=512)
    parser.add_argument("--worker-python", type=Path,
                        help="Python with langgraph-cli installed; defaults to this interpreter")
    return parser


if __name__ == "__main__":
    cli(main, parser=create_parser())
