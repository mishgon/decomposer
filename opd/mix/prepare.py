"""Prepare the initial 100% WideSeek-depth mixture with the existing data contract."""
import argparse
import hashlib
import json
from pathlib import Path

from opd.wideseek.prepare import prepare as prepare_wideseek

RECIPE = Path(__file__).with_name('recipe.json')


def prepare(source, output):
    raw_recipe = RECIPE.read_bytes()
    recipe = json.loads(raw_recipe)
    if recipe != {'training_sources': [{'gym': 'wideseek', 'split': 'depth', 'weight': 1.0}]}:
        raise ValueError('Only the 100% WideSeek-depth mixture is implemented')
    with source.open() as stream:
        if any(json.loads(line)['source'] != 'depth' for line in stream):
            raise ValueError('The initial OPD mix requires WideSeek depth tasks only')
    manifest = prepare_wideseek(source, output)
    (output / 'mixture.json').write_bytes(raw_recipe)
    manifest['training_mixture'] = recipe['training_sources']
    manifest['files']['mixture.json'] = hashlib.sha256(raw_recipe).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
