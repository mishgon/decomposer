"""Collect DeepDive QA traces; retain raw failures and index correct answers."""
from sft.deepdive.prepare import DEFAULT_OUTPUT
from gyms.wideseek.run import cli
from sft.wideseek.run import create_collection_parser as shared_parser, main


def create_collection_parser():
    parser = shared_parser()
    parser.description = __doc__
    parser.set_defaults(data=DEFAULT_OUTPUT / "tasks.jsonl", limit=None, concurrency=128)
    return parser


if __name__ == "__main__":
    cli(main, parser=create_collection_parser())
