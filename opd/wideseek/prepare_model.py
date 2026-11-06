"""Rebuild the pinned SFT v1.2.0 release on the shared disk and verify its hash."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO = 'decomposer-datasets/decomposer-manager-qwen35-4b-sft'
REVISION = '70f7732e58f25be7787d7ff75761ef85fe90c98e'


def main(args):
    from huggingface_hub import snapshot_download
    token = (Path.home() / '.config/huggingface/token').read_text().strip()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = Path(snapshot_download(REPO, revision=REVISION, token=token,
                                   allow_patterns=['merge.py', '1.2.0/*']))
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
                    'python', str(source / 'merge.py'), '1.2.0', str(args.output.resolve()),
                    '--source', str(source / '1.2.0')], env=env, check=True)
    # merge.py exits unsuccessfully if this does not match the published hash.
    weights = args.output / 'model.safetensors'
    with weights.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    expected = (source / '1.2.0/merged/model.safetensors.sha256').read_text().split()[0]
    assert digest == expected
    (args.output / 'opd_initialization.json').write_text(json.dumps(
        {'repo': REPO, 'tag': 'v1.2.0', 'revision': REVISION, 'sha256': digest}, indent=2))
    print(f'Verified SFT v1.2.0: {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    main(parser.parse_args())
