"""Collect a fresh teacher coverage pass with embedded ClearML reporting.

Pass a normalized, shuffled task JSONL combining WideSeek and DeepDive sources.
All models are hosted; this launcher needs no local inference server or GPU.
"""
import json
import subprocess
import time

from evals.wideseek.clearml_reporter import collection_reporter
from gyms.wideseek import REPO_ROOT
from gyms.wideseek.run import cli
from gyms.wideseek.runtime import save
from sft.wideseek.launch import create_parser, main as collect


def main():
    parser = create_parser()
    parser.description = __doc__
    parser.set_defaults(limit=None, concurrency=128, n=1)
    parser.add_argument('--clearml-project', default='decomposer-coverage')
    parser.add_argument('--clearml-name')
    args = parser.parse_args()
    if args.agent != 'decomposer' or args.adaptive:
        parser.error('Coverage uses --agent decomposer and fixed attempts, without --adaptive')
    root = args.output.resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'launch.json'
    if path.exists() and not args.resume:
        parser.error('Run already exists; use --resume to preserve completed results')
    state = json.loads(path.read_text()) if path.exists() else {
        'kind': 'coverage', 'started_at': time.time(),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip(),
        'concurrency': args.concurrency, 'repetitions': args.n,
    }
    state['status'] = 'running'
    save(path, state)
    with collection_reporter(args):
        try:
            cli(collect, parser=parser)
            state['status'] = 'completed'
        except BaseException as exc:
            state.update(status='interrupted', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            state['finished_at'] = time.time()
            save(path, state)


if __name__ == '__main__':
    main()
