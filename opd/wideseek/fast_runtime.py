"""Isolated, verified Qwen3.5 kernel overlay restored from the Hertz-2 ablation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


DIRECTORIES = ('triton-packages', 'packages', 'conv-packages')
VERSIONS = {'torch': '2.10.0', 'transformers': '5.5.4', 'triton': '3.7.1',
            'flash-linear-attention': '0.5.2', 'fla-core': '0.5.2', 'causal-conv1d': '1.7.0'}


def checksums(root):
    result = {}
    for directory in DIRECTORIES:
        for path in sorted((root / directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                with path.open('rb') as stream:
                    result[str(path.relative_to(root))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def restore(source, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in DIRECTORIES:
        shutil.copytree(source / name, output / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (output / 'manifest.json').write_text(json.dumps({
        'source': str(source.resolve()), 'python': '3.12', 'versions': VERSIONS,
        'files': checksums(output)}, indent=2))


def environment(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['versions'] != VERSIONS or manifest['python'] != '3.12':
        raise ValueError('Unexpected fast-runtime profile')
    if checksums(root) != manifest['files']:
        raise ValueError('Fast-runtime bundle contents changed')
    return {**os.environ, 'FLA_TILELANG': '0',
            'PYTHONPATH': ':'.join([*(str(root / n) for n in DIRECTORIES), os.environ.get('PYTHONPATH', '')])}


def check():
    from importlib.metadata import version
    import torch
    if torch.__version__ != '2.10.0+cu128' or torch.version.cuda != '12.8':
        raise ValueError('Fast runtime requires the Torch 2.10 CUDA 12.8 build')
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Fast runtime requires Python 3.12')
    for package, expected in VERSIONS.items():
        if version(package) != expected:
            raise ValueError(f'{package}: expected {expected}, found {version(package)}')
    import causal_conv1d
    from transformers.models.qwen3_5 import modeling_qwen3_5 as model
    for name in ('chunk_gated_delta_rule', 'FusedRMSNormGated', 'causal_conv1d_fn'):
        if not callable(getattr(model, name, None)):
            raise RuntimeError(f'Qwen3.5 fast path unavailable: {name}')
    if os.environ.get('FLA_TILELANG') != '0':
        raise ValueError('This profile uses Triton FLA, not TileLang')
    print(json.dumps({'versions': VERSIONS, 'fast_path': 'ready'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-from', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.restore_from:
        if not args.output:
            parser.error('--restore-from requires --output')
        restore(args.restore_from, args.output)
    else:
        check()
