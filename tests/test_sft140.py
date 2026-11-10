import json

import pytest

from evals.sft140 import MODELS, jobs
from gyms.browsecomp.run import parser as browsecomp_parser
from gyms.widesearch.run import parser as widesearch_parser
from gyms.gaia2.run import build_parser


def test_campaign_schedule_and_commands(tmp_path):
    data = tmp_path / 'tasks.jsonl'
    data.write_text('\n'.join(json.dumps({'task_id': f'browsecomp-{i:04d}'}) for i in range(100)))
    schedule = jobs(data, {'execution': 'execution_task', 'search': 'search_task'})
    assert len(schedule) == 25
    assert len({j['name'] for j in schedule}) == 25
    assert [j['benchmark'] for j in schedule[13:]] == ['browsecomp'] * 5 + ['widesearch'] * 5 + ['gaia2'] * 2
    for job in schedule:
        command = job['command'][2:]
        if job['benchmark'] == 'gaia2':
            args = build_parser(evaluation=True).parse_args(command)
            assert args.agent_run_budget_seconds == args.agent_shutdown_grace_seconds == 60
            assert args.agent_model_id == 'vllm/qwen_3_5_4b_thinking'
            assert args.num_repeats == (1 if job['name'].startswith('smoke/') else 3)
        else:
            parser = browsecomp_parser() if job['benchmark'] == 'browsecomp' else widesearch_parser()
            args = parser.parse_args(command)
            assert args.agent in MODELS
            assert args.concurrency == (2 if job['benchmark'] == 'browsecomp' and not job['name'].startswith('smoke/') else 1)
            assert args.repetitions == (1 if job['name'].startswith('smoke/') else 3)
    pilot = browsecomp_parser().parse_args(schedule[12]['command'][2:])
    assert pilot.tasks == [f'browsecomp-{i:04d}' for i in range(5)]


def test_campaign_rejects_wrong_selection(tmp_path):
    data = tmp_path / 'tasks.jsonl'
    data.write_text(json.dumps({'task_id': 'a'}))
    with pytest.raises(ValueError, match='100-task'):
        jobs(data, {})
