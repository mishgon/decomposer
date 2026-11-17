"""Run a fresh checkpoint against the frozen seen-task pool on one GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from gyms.wideseek import REPO_ROOT


def prepare_pool(source, root, recipe):
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != recipe['pool_sha256']:
        raise ValueError('Seen task pool differs from the frozen recipe')
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != recipe['pool_tasks'] or len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('Seen task pool has an unexpected task count or duplicate IDs')
    target = root / 'tasks.jsonl'
    if target.exists() and target.read_bytes() != data:
        raise ValueError('Cannot replace the task pool of an existing run')
    target.write_bytes(data)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--inference-python', type=Path, required=True)
    parser.add_argument('--sft-version', help='Override the recipe checkpoint for verification')
    args = parser.parse_args()
    recipe = json.loads(Path(__file__).with_name('seen.json').read_text())
    if args.sft_version:
        recipe['sft_version'] = args.sft_version.removeprefix('v')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    recipe_path = root / 'recipe.json'
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError('Cannot change checkpoint or budgets in an existing run directory')
    if (root / 'eval').exists() and not recipe_path.exists():
        raise ValueError('Use a new run directory for the 15-minute recipe')
    data = prepare_pool(args.data, root, recipe)
    recipe_path.write_text(json.dumps(recipe, indent=2) + '\n')
    model = args.models.resolve() / f"decomposer-manager-sft-v{recipe['sft_version']}"
    if not (model / 'opd_initialization.json').exists():
        subprocess.run([str(args.inference_python), '-m', 'opd.wideseek.prepare_model',
                        '--gpu', '0', '--version', recipe['sft_version'], '--output', str(model)], check=True)
    initialization = json.loads((model / 'opd_initialization.json').read_text())
    if initialization['tag'] != f"v{recipe['sft_version']}":
        raise ValueError('Cached checkpoint does not match the requested SFT version')
    with (model / 'model.safetensors').open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != initialization['sha256']:
            raise ValueError('Cached checkpoint checksum mismatch')
    print(f"Verified SFT {initialization['tag']}; {recipe['pool_tasks']} tasks, "
          f"{recipe['repetitions']} attempts, concurrency {recipe['concurrency']}, "
          f"episode budget {recipe['episode_timeout']}s", flush=True)
    command = [sys.executable, '-u', str(REPO_ROOT / 'scripts/run-inband.py'),
               '--model', str(model), '--inference-python', str(args.inference_python),
               '--data', str(data), '--output', str(root / 'eval'), '--agent', 'student',
               '--limit', str(recipe['pool_tasks']), '-n', str(recipe['repetitions']),
               '--concurrency', str(recipe['concurrency']),
               '--episode-timeout', str(recipe['episode_timeout']),
               '--success-threshold', str(recipe['success_threshold']),
               '--max-pass-rate', str(recipe['max_pass_rate']),
               '--worker-port', '0', '--worker-slots', '1024', '--worker-python', sys.executable,
               '--clearml-name', root.name]
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
