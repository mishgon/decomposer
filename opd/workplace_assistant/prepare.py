"""Resolve the frozen Workplace IDs once, before policy validation starts."""
import json
from pathlib import Path


def validation_rows():
    from datasets import load_dataset
    validation = []
    task_ids = json.loads((Path(__file__).resolve().parents[2] /
                           'gyms/workplace_assistant/validation20.json').read_text())
    dataset = load_dataset('nvidia/Nemotron-RL-agent-workplace_assistant',
        revision='f674d9bba17820d90f14eefe9156fdc018a0db60', split='validation')
    by_id = {row['id']: row for row in dataset}
    for task_id in task_ids:
        row = by_id[task_id]
        task = {'id': task_id, 'prompt': row['responses_create_params']['input'][-1]['content'],
                'ground_truth': row['ground_truth'], 'environment_name': row['environment_name']}
        validation.append({'data_source': 'workplace-assistant/validation20',
            'agent_name': 'workplace_decomposer',
            'prompt': [{'role': 'user', 'content': task['prompt']}],
            'reward_model': {'style': 'rule', 'ground_truth': str(task['id'])},
            'extra_info': {'task_id': str(task['id']), 'task': json.dumps(task)}})
    return validation
