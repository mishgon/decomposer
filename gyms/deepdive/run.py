"""Run DeepDive QA with WideSeek agents, offline tools, raw traces and QA judging."""
from gyms.deepdive import DEFAULT_DATA
from gyms.wideseek.run import cli, create_parser as shared_parser, main


def create_parser():
    parser = shared_parser()
    parser.description = __doc__
    parser.set_defaults(data=DEFAULT_DATA, n=1)
    return parser


if __name__ == "__main__":
    cli(main, parser=create_parser())
