"""Live terminal view of the bounded WideSeek actor benchmark."""
import argparse
import json
from pathlib import Path
import sys
import time


def display(root):
    path = root / 'status.json'
    if not path.exists():
        return f'WIDESEEK ACTOR BENCHMARK\nWaiting for startup: {root}'
    state = json.loads(path.read_text())
    seconds = state.get('finished_at', time.time()) - state['started_at']
    lines = [f"WIDESEEK ACTOR BENCHMARK | {state['status'].upper()} | GPU {state['gpu']}",
             f"{state['completed']}/{state['total']} variants | elapsed {seconds / 60:.1f}m | active: {state['active']}",
             'ETA: unknown until kernel compilation completes' if state['status'] == 'running' else 'Measurements finished', '',
             'VARIANT          TOKENS   STEP TIME   PEAK VRAM   GRAD COS   REL ERROR   CHECK']
    for row in state['results']:
        cosine = f"{row['gradient_cosine']:.6f}" if 'gradient_cosine' in row else '--'
        error = f"{row['relative_gradient_error']:.2%}" if 'relative_gradient_error' in row else '--'
        check = ('PASS' if row['numerical_check'] else 'FAIL') if 'numerical_check' in row else 'reference'
        lines.append(f"{row['variant']:<16} {row['length']:>6}   {row['median_seconds']:>7.2f}s   "
                     f"{row['peak_gib']:>6.1f}GiB   {cosine:>8}   {error:>9}   {check}")
    active = root / f"{state['active']}.json"
    if state['status'] == 'running' and active.exists():
        rows = json.loads(active.read_text())['steps']
        lines += ['', f"Active variant: {len(rows)}/3 passes saved (first pass is warmup)."]
    if state.get('error'):
        lines += ['', state['error']]
    lines.extend(['', *state.get('notes', [])])
    return '\n'.join(lines + ['', 'Fixed tokens; no teacher requests or training-job launch. Ctrl-C closes this watcher only.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    try:
        while True:
            if sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print(display(args.root), flush=True)
            if args.once:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        pass
