"""Run the initial 100% WideSeek-depth OPD mixture after its checkpoint baseline."""
import asyncio

from opd.mix.prepare import prepare
from opd.wideseek.run import create_parser, main as run

OVERRIDES = ['trainer.project_name=decomposer-mix-opd', 'wideseek.timeout=2700']


async def main(args, overrides):
    await run(args, [*OVERRIDES, *overrides],
              prepare_data=prepare)


if __name__ == '__main__':
    parser = create_parser()
    parser.description = __doc__
    args, overrides = parser.parse_known_args()
    asyncio.run(main(args, overrides))
