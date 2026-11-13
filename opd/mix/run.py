"""Run the initial 100% WideSeek-depth OPD mixture after its checkpoint baseline."""
import asyncio

from opd.mix.prepare import prepare
from opd.wideseek.run import create_parser, main as run


async def main(args, overrides):
    await run(args, ['trainer.project_name=decomposer-mix-opd', *overrides],
              prepare_data=prepare)


if __name__ == '__main__':
    parser = create_parser()
    parser.description = __doc__
    args, overrides = parser.parse_known_args()
    asyncio.run(main(args, overrides))
