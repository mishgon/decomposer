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


@pytest.mark.parametrize("gaia_error", [None, "TimeoutError", "GraphRecursionError", "APITimeoutError", "ReadTimeout", "ConnectTimeout"])
def test_queue_completes_and_resumes_without_rerunning(monkeypatch, tmp_path, gaia_error):
    from types import SimpleNamespace
    from evals import sft140
    from gyms.gaia2 import run as gaia
    data = tmp_path / 'tasks.jsonl'
    data.write_text('selection')
    gaia_manifest = tmp_path / 'gaia.json'
    gaia_manifest.write_text(json.dumps({'scenarios': [{'scenario_id': str(i)} for i in range(160)]}))
    monkeypatch.setattr(gaia, 'dataset_manifest', lambda _: gaia_manifest)
    monkeypatch.setattr(sft140, 'jobs', lambda *_: [
        {'name': 'full/browsecomp/test', 'benchmark': 'browsecomp', 'command': ['-m', 'evals.browsecomp.run']},
        {'name': 'full/gaia2/test', 'benchmark': 'gaia2', 'domain': 'search', 'command': ['-m', 'evals.gaia2.run']},
    ])
    async def preflight(_):
        return {'ready': True}
    monkeypatch.setattr(sft140, 'preflight', preflight)
    monkeypatch.setattr(sft140.subprocess, 'check_output', lambda *a, **kw: 'fixed-revision')
    spawned = []
    def spawn(command, **kwargs):
        spawned.append(command)
        directory = sft140.Path(command[command.index('--output-dir') + 1])
        if '--resume' in command:
            prior = sft140.Path(command[command.index('--resume') + 1])
            assert json.loads((prior / 'manifest.json').read_text())['status'] == 'interrupted'
        if 'evals.gaia2.run' in command:
            target = directory / 'search/eval_metrics.json'
            value = {'gym_metrics': {'exception_types': {gaia_error: 1} if gaia_error else {}}}
        else:
            target = directory / 'run/manifest.json'
            value = {'status': 'completed'}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
        return SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(sft140.subprocess, 'Popen', spawn)
    args = SimpleNamespace(output_dir=tmp_path / 'results', browsecomp_data=data, old_browsecomp_run=tmp_path)
    if gaia_error in {'APITimeoutError', 'ReadTimeout', 'ConnectTimeout'}:
        with pytest.raises(RuntimeError, match='infrastructure errors'):
            sft140.run(args)
        state = json.loads((args.output_dir / 'queue.json').read_text())
        assert state['status'] == 'needs_attention'
        assert 'full/gaia2/test' not in state['completed']
        return
    sft140.run(args)
    sft140.run(args)
    state = json.loads((args.output_dir / 'queue.json').read_text())
    assert state['status'] == 'completed'
    assert len(spawned) == len(state['completed']) == 2

    state.update(status='running', child_pid=99999999, completed=['full/gaia2/test'])
    (args.output_dir / 'queue.json').write_text(json.dumps(state))
    stale = args.output_dir / 'full/browsecomp/test/run/manifest.json'
    stale.write_text(json.dumps({'status': 'running'}))
    def dead_pid(*args):
        raise ProcessLookupError()
    monkeypatch.setattr(sft140.os, 'kill', dead_pid)
    sft140.run(args)
    assert len(spawned) == 3
    assert '--resume' in spawned[-1]


def test_pilot_counts_recorded_timeout_status(tmp_path):
    from evals.sft140 import pilot_report
    old = tmp_path / 'old'
    new = tmp_path / 'pilot/browsecomp/decomposer_sft_1_4_0/new'
    for path in [old, new]:
        (path / 'trace').mkdir(parents=True)
        (path / 'manifest.json').write_text(json.dumps({'tasks': ['a'], 'episodes': [
            {'task': 'a', 'directory': 'trace', 'evaluation': {'passed': False}}]}))
        (path / 'trace/trace.json').write_text(json.dumps({'agent_runs': {
            'worker': {'status': 'timeout', 'error': None}}}))
    pilot_report(tmp_path, old)
    report = json.loads((tmp_path / 'pilot-comparison.json').read_text())
    assert report['old_60_60']['worker_timeouts'] == 1
    assert report['new_180_180']['worker_timeouts'] == 1
