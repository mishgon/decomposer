"""Launch DeepDive collection with dedicated local researcher workers."""
from pathlib import Path

from gyms.deepdive import REPO_ROOT
from gyms.wideseek.run import cli
from sft.deepdive.run import create_collection_parser, main as collect
from sft.wideseek.launch import run_with_workers


async def main(args):
    await run_with_workers(args, REPO_ROOT / "gyms/deepdive/langgraph.json", collect)


def create_parser():
    parser = create_collection_parser()
    parser.description = __doc__
    parser.add_argument("--worker-port", type=int, default=0,
                        help="Worker server port (default: automatically allocated on each start)")
    parser.add_argument("--worker-slots", type=int, default=512)
    parser.add_argument("--worker-python", type=Path,
                        help="Python with langgraph-cli installed; defaults to this interpreter")
    return parser


if __name__ == "__main__":
    cli(main, parser=create_parser())
