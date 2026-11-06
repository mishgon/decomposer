"""Screen native validation tasks before model calibration; never freeze by score here."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from gyms.workplace_assistant.in_process import SYSTEM, TOOLKITS
from resources_servers.workplace_assistant.utils import get_tools, is_correct


def prompt(row):
    return row['responses_create_params']['input'][-1]['content']


def normalized(text):
    return ' '.join(text.casefold().split())


def screen(train, validation):
    training = {normalized(prompt(row)) for row in train}
    seen = set()
    accepted = []
    rejected = Counter()
    for row in validation:
        text = prompt(row)
        key = normalized(text)
        if key in training or key in seen:
            rejected['overlap_or_duplicate'] += 1
            continue
        seen.add(key)
        reference = row['ground_truth']
        if not reference or is_correct([], reference, ''):
            rejected['no_action_passes'] += 1
            continue
        try:
            env = get_tools(TOOLKITS)
            for action in reference:
                env['functions'][action['name']](**json.loads(action['arguments']))
        except Exception:
            rejected['reference_exception'] += 1
            continue
        assert is_correct(reference, reference, '')
        assert row['responses_create_params']['input'][0]['content'] == SYSTEM
        accepted.append({'id': row['id'], 'prompt': text, 'ground_truth': reference,
                         'environment_name': row['environment_name'], 'category': row['category'],
                         'prompt_sha256': hashlib.sha256(text.encode()).hexdigest()})
    return accepted, dict(rejected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Optional pilot size; default screens the full split')
    args = parser.parse_args()
    accepted, rejected = screen(json.loads((args.source/'train.json').read_text()),
                                json.loads((args.source/'validation.json').read_text()))
    # Round-robin domains, with short prompts and few reference mutations first.
    groups = defaultdict(list)
    for row in sorted(accepted, key=lambda r: (len(r['ground_truth']), len(r['prompt']), r['id'])):
        groups[row['category']].append(row)
    selected = []
    limit = args.limit if args.limit is not None else len(accepted)
    while any(groups.values()) and len(selected) < limit:
        for key in sorted(groups):
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0))
    args.output.write_text(json.dumps(selected, indent=2, ensure_ascii=False))
    report = {'eligible': len(accepted), 'selected': len(selected), 'rejected': rejected,
              'source': json.loads((args.source/'source.json').read_text()),
              'split': 'validation', 'overlap_check': 'normalized prompts against source train split'}
    args.output.with_suffix('.audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
