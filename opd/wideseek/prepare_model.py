"""Merge an SFT release on the shared disk and verify its published hash."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

REPO = 'decomposer-datasets/decomposer-manager-qwen35-4b-sft'
REVISION = 'ac9747fd7dff94201bd82faf393cdf64f5de3e80'


def main(args):
    from huggingface_hub import HfApi, snapshot_download
    version = args.version.removeprefix('v')
    if not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('Expected an SFT release version such as 1.8.0')
    token = os.environ.get('HF_TOKEN') or (Path.home() / '.config/huggingface/token').read_text().strip()
    if args.output.exists():
        raise FileExistsError(args.output)
    revision = REVISION if version == '1.7.0' else HfApi(token=token).model_info(REPO).sha
    source = Path(snapshot_download(REPO, revision=revision, token=token,
                                   allow_patterns=['merge.py', f'{version}/*']))
    expected = (source / version / 'merged/model.safetensors.sha256').read_text().split()[0]
    used = int(subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
               '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if used > 1024:
        raise RuntimeError(f'GPU {args.gpu} is occupied ({used} MiB)')
    env = dict(os.environ, HF_TOKEN=token, CUDA_VISIBLE_DEVICES=str(args.gpu))
    # The release requires these versions for byte-identical GPU merging.
    # uv keeps this separate from the pinned OPD environment.
    subprocess.run(['uv', 'run', '--no-project', '--python', '3.12',
                    '--with', 'torch==2.11.0', '--with', 'transformers==5.14.1',
                    '--with', 'peft==0.21.2', '--with', 'huggingface_hub',
                    'python', str(source / 'merge.py'), version, str(args.output.resolve()),
                    '--source', str(source / version)], env=env, check=True)
    # merge.py exits unsuccessfully if this does not match the published hash.
    weights = args.output / 'model.safetensors'
    with weights.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == expected
    (args.output / 'opd_initialization.json').write_text(json.dumps(
        {'repo': REPO, 'tag': f'v{version}', 'revision': revision, 'sha256': digest}, indent=2))
    print(f'Verified SFT v{version}: {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--version', default='1.7.0')
    main(parser.parse_args())
