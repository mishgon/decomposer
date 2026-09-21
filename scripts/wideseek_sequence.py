"""Run separate WideSeek setups sequentially; stop the queue on process failure."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time

from gyms.wideseek.runtime import save


def main(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs = [f"{args.name}-{mode}" for mode in ("simple", "decomposer")]
    if any((root / name).exists() for name in jobs):
        raise ValueError("Use a fresh sequence name; existing runs are never overwritten")
    state = {"jobs": jobs, "completed": [], "active": None, "pending": jobs.copy(),
             "status": "running", "started_at": time.time()}
    with (root / 'sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for mode, name in zip(("simple", "decomposer"), jobs):
            state.update(active=name, pending=state['pending'][1:])
            save(root / 'sequence.json', state)
            with (root / f'{name}.log').open('x') as log:
                result = subprocess.run([sys.executable, '-m', 'gyms.wideseek.run',
                    '--mode', mode, '--output', str(root / name), '--limit', str(args.limit),
                    '-n', str(args.n), '--concurrency', str(args.concurrency)], stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                state.update(status='failed', exit_code=result.returncode)
                save(root / 'sequence.json', state)
                return result.returncode
            state['completed'].append(name)
        state.update(status='completed', active=None, finished_at=time.time())
        save(root / 'sequence.json', state)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('artifacts/gyms/wideseek/runs'))
    p.add_argument('--name', required=True)
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('-n', type=int, default=3)
    p.add_argument('--concurrency', type=int, default=2)
    sys.exit(main(p.parse_args()))
