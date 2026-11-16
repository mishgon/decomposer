import pytest

from evals.wideseek.clearml_reporter import prefix_cache_metrics, snapshot, timeline
from gyms.wideseek.runtime import save


def result(task, score, end, **extra):
    return {'task_id': task, 'started_at': end - 60, 'finished_at': end,
            'status': 'finished', 'evaluation': {'status': 'scored', 'score': score}, **extra}


def manifest(tasks):
    return {'started_at': 1000, 'settings': {'tasks': tasks, 'repetitions': 4, 'modes': ['decomposer']}}


def test_prefix_cache_uses_interval_token_counts_across_engines():
    before = '''
vllm:prefix_cache_queries_total{engine="0"} 100
vllm:prefix_cache_hits_total{engine="0"} 80
vllm:prefix_cache_queries_total{engine="1"} 100
vllm:prefix_cache_hits_total{engine="1"} 20
'''
    values, counters = prefix_cache_metrics(before, None)
    assert values == {}
    after = before.replace(' 100', ' 200').replace(' 80', ' 180').replace(' 20\n', ' 70\n')
    values, current = prefix_cache_metrics(after, counters)
    assert values == {('vLLM prefix cache (%)', 'Hit (last interval)'): 75,
                      ('vLLM prefix cache (%)', 'Miss (last interval)'): 25}
    assert prefix_cache_metrics(after, current)[0] == {}  # Idle is not 0% hits.
    assert prefix_cache_metrics(before, current)[0] == {}  # Counter reset.
    assert prefix_cache_metrics('', counters) == ({}, None)


def test_four_attempt_selection_matches_native_scoring_and_excludes_infra():
    rows = []
    for passes in range(5):
        rows.extend(result(f'depth-{passes}', int(i < passes), 1060 + i) for i in range(4))
    rows.extend(result('width-unknown', 0, 1100 + i) for i in range(3))
    rows.append(result('width-unknown', None, 1104, evaluation={'status': 'evaluation_error'}))
    _, values = timeline(rows, manifest([f'depth-{i}' for i in range(5)] + ['width-unknown']), .9, .5)
    assert values['In-band tasks', 'All'] == 3
    assert values['In-band tasks', 'WideSeek depth'] == 3
    assert values['In-band tasks', 'WideSeek width'] == 0
    assert values['Episodes', 'Resolved'] == 23
    assert values['Episodes', 'Unresolved'] == 1
    assert values['Evaluated tasks', 'All attempts resolved'] == 5
    assert values['Pass rate (%)', 'Resolved episodes'] == pytest.approx(100 * 10 / 23)
    assert values['Native score', 'Mean scored episodes'] == pytest.approx(10 / 23)
    for i in range(5):
        assert values['Successes per evaluated task', f'{i}/4'] == 1
    _, unseen = timeline(rows, manifest([f'depth-{i}' for i in range(5)] + ['width-unknown']),
                         .9, .5, require_success=True)
    assert unseen['In-band tasks', 'All'] == 2
    assert unseen['Episodes', 'Unresolved'] == 1


def test_backfill_orders_results_and_does_not_count_imported_results_in_throughput():
    rows = [result('depth-0', 1, 1121), result('depth-0', 0, 900),
            result('depth-0', 0, 1061), result('depth-0', 0, 1120)]
    points, _ = timeline(rows, manifest(['depth-0']), .9, .5)
    assert list(points) == [0, 1, 2]
    assert [p['Episodes', 'Completed'] for p in points.values()] == [1, 2, 4]
    assert points[2]['In-band tasks', 'All'] == 1
    assert points[2]['Throughput (episodes/hour)', 'Since collection start'] == 60


def test_snapshot_requires_captured_trace_and_leaves_run_unchanged(tmp_path):
    save(tmp_path / 'manifest.json', manifest(['depth-0']))
    attempt = tmp_path / 'decomposer/depth-0/attempt-001'
    save(attempt / 'result.json', result('depth-0', 1, 1060, execution_directory='execution'))
    (tmp_path / 'decomposer/depth-0/attempt-002').mkdir(parents=True)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    m, rows, unfinished = snapshot(tmp_path)
    assert unfinished == 1
    _, values = timeline(rows, m, .9, .5)
    assert values['Episodes', 'Unresolved'] == 1
    assert values['Episodes', 'Resolved'] == 0
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_expected_timeout_is_zero_pass_but_not_a_fake_native_score():
    rows = [result('deepdive-qa_rl-1', None, 1060, status='timeout', evaluation={'status': 'unscored'})]
    _, values = timeline(rows, manifest(['deepdive-qa_rl-1']), .9, .5)
    assert values['Episodes', 'Resolved'] == 1
    assert values['Pass rate (%)', 'Resolved episodes'] == 0
    assert ('Native score', 'Mean scored episodes') not in values


def test_embedded_reporter_flushes_after_collection_finishes(tmp_path, monkeypatch):
    import json
    from threading import Event
    from types import SimpleNamespace
    from evals.wideseek import clearml_reporter as reporter

    run = tmp_path / 'eval'
    save(run / 'manifest.json', {})
    save(tmp_path / 'launch.json', {'status': 'running'})
    ready = Event()
    observed = []

    def report(options, stop):
        assert options.run == run
        assert options.state == tmp_path / 'clearml.json'
        assert options.max_pass_rate == .5
        ready.set()
        assert stop.wait(5)
        observed.append(json.loads((tmp_path / 'launch.json').read_text())['status'])

    monkeypatch.setattr(reporter, 'report', report)
    args = SimpleNamespace(output=run, clearml_project='test', clearml_name='example',
                           success_threshold=.9, max_pass_rate=.5)
    with pytest.raises(RuntimeError, match='collection failure'):
        with reporter.collection_reporter(args):
            assert ready.wait(5)
            save(tmp_path / 'launch.json', {'status': 'interrupted'})
            raise RuntimeError('collection failure')
    assert observed == ['interrupted']


def test_reporter_failure_does_not_cancel_collection(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from evals.wideseek import clearml_reporter as reporter

    save(tmp_path / 'eval/manifest.json', {})
    def fail(*args):
        raise ConnectionError('unavailable')
    monkeypatch.setattr(reporter, 'report', fail)
    args = SimpleNamespace(output=tmp_path / 'eval', clearml_project='test', clearml_name=None,
                           success_threshold=.9, max_pass_rate=.5)
    with pytest.warns(UserWarning, match='collection continues'):
        with reporter.collection_reporter(args):
            pass


def test_startup_failure_does_not_create_clearml_task(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from evals.wideseek import clearml_reporter as reporter

    report = Mock()
    monkeypatch.setattr(reporter, 'report', report)
    args = SimpleNamespace(output=tmp_path / 'eval', clearml_project='test', clearml_name=None,
                           success_threshold=.9, max_pass_rate=.5)
    with reporter.collection_reporter(args):
        pass
    report.assert_not_called()


def test_report_flushes_final_metrics_and_reuses_task_on_resume(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    from evals.wideseek import clearml_reporter as reporter

    run = tmp_path / 'eval'
    data = manifest(['depth-0'])
    data['revision'] = 'collector-commit'
    data['settings'].update(concurrency=128, episode_timeout=2700, model_profiles={})
    save(run / 'manifest.json', data)
    save(tmp_path / 'launch.json', {'status': 'completed'})
    task = Mock(id='existing-task')
    task.get_output_log_web_page.return_value = 'https://clearml.example/task'
    task_type = SimpleNamespace(init=Mock(return_value=task), TaskTypes=SimpleNamespace(testing='testing'))
    monkeypatch.setitem(sys.modules, 'clearml', SimpleNamespace(Task=task_type))
    args = SimpleNamespace(run=run, state=tmp_path / 'clearml.json', project='test', name='run',
                           success_threshold=.9, max_pass_rate=.5, interval=60)
    reporter.report(args)
    reporter.report(args)
    assert task_type.init.call_args.kwargs['reuse_last_task_id'] == 'existing-task'
    assert task_type.init.call_args.kwargs['continue_last_task'] == 0
    assert task.flush.call_count == 2
    task.flush.assert_called_with(wait_for_uploads=True)
    assert task.close.call_count == 2
    task.mark_stopped.assert_not_called()
    assert json.loads(args.state.read_text())['collection_status'] == 'completed'
    assert any(call.args[:3] == ('Episodes', 'Active', 0)
               for call in task.get_logger.return_value.report_scalar.call_args_list)


@pytest.mark.parametrize('stale', [False, True])
def test_gpu_telemetry_selects_collector_assignment_and_rejects_stale_samples(monkeypatch, stale):
    from datetime import datetime, timezone
    import httpx
    from evals.wideseek import clearml_reporter as reporter

    monkeypatch.setenv('OCC_TOKEN', 'test-token')
    monkeypatch.setattr(reporter.time, 'time', lambda: 1000)
    def handler(request):
        assert request.headers['Authorization'] == 'Bearer test-token'
        if request.url.path.endswith('/attempts'):
            return httpx.Response(200, json={'items': [
                {'attemptNumber': 1, 'status': 'Failed', 'workers': []},
                {'attemptNumber': 2, 'status': 'Running', 'workers': [
                    {'nodeId': 'node', 'workerIndex': 0, 'gpuDeviceIds': ['ours']}]}]})
        assert request.url.path == '/api/v1/nodes/node'
        return httpx.Response(200, json={'inventory': {
            'observedAt': datetime.fromtimestamp(700 if stale else 999, timezone.utc).isoformat(),
            'gpus': [{'id': 'other', 'index': 0, 'utilizationPercent': 100, 'memoryUsedBytes': 0},
                     {'id': 'ours', 'index': 3, 'utilizationPercent': 40, 'memoryUsedBytes': 2 * 1024 ** 3}]}})
    client = httpx.Client
    monkeypatch.setattr(reporter.httpx, 'Client', lambda **kw: client(transport=httpx.MockTransport(handler), **kw))
    values = reporter.collector_gpu_metrics('https://scheduler.example/api/v1', 'job')
    assert values == ({} if stale else {
        ('GPU utilization (%)', 'Worker 0 GPU 3'): 40,
        ('GPU memory (GiB)', 'Worker 0 GPU 3'): 2})


def test_subagent_statistics_count_successful_new_and_terminal_overlap():
    from gyms.wideseek.metrics import episode_subagent_statistics
    messages = [{'type': 'ai', 'tool_calls': [
        {'id': str(i), 'name': name} for i, name in enumerate(['new', 'new', 'new', 'fork', 'run'])]}]
    messages += [{'type': 'tool', 'tool_call_id': str(i), 'content': content} for i, content in enumerate([
        '{"agent_id":"a"}', '{"agent_id":"b"}', '{"error":"failed"}',
        '{"agent_id":"c"}', '{"agent_run_id":"r"}'])]
    messages.append(messages[1])  # A repeated serialized result is not another invocation.
    trace = {'messages': messages, 'agents': {x: {'thread_id': x} for x in 'abc'},
             'agent_runs': {'r': {'run_id': 'r'}}}
    def run(id, start, end):
        return {'run_id': id, 'status': 'success',
                'created_at': f'2026-11-16T00:00:{start:02d}+00:00',
                'updated_at': f'2026-11-16T00:00:{end:02d}+00:00'}
    archives = {'a': {'runs': [run('r', 0, 10), run('r2', 20, 30)]},
                'b': {'runs': [run('r3', 5, 15)]}, 'c': {'runs': [run('r4', 15, 20)]}}
    assert episode_subagent_statistics(trace, archives) == {'created': 2, 'peak_concurrency': 2, 'no_delegation': 0}
    # Missing cleanup archives must not produce an artificially low peak.
    del archives['b']
    assert episode_subagent_statistics(trace, archives) == {'created': 2, 'no_delegation': 0}
    assert episode_subagent_statistics({'messages': [], 'agents': {}, 'agent_runs': {}}, {}) == {
        'created': 0, 'peak_concurrency': 0, 'no_delegation': 1}
    assert episode_subagent_statistics({}, {}) == {}


@pytest.mark.parametrize('tool', ['new', 'fork', 'run'])
def test_zero_delegation_excludes_even_failed_or_unanswered_attempts(tool):
    from gyms.wideseek.metrics import episode_subagent_statistics
    trace = {'messages': [{'type': 'ai', 'tool_calls': [{'id': 'a', 'name': tool}]}]}
    assert episode_subagent_statistics(trace, {})['no_delegation'] == 0


def test_no_delegation_counter_only_counts_successful_captured_episodes():
    rows = [result('depth-a', 1, 1060, agent_statistics={'no_delegation': 1}),
            result('depth-b', 0, 1061, agent_statistics={'no_delegation': 1}),
            result('width-c', .9, 1062, agent_statistics={'no_delegation': 1}),
            result('deepdive-qa_rl-d', 1, 1063, agent_statistics={'no_delegation': 0}),
            result('deepdive-qa_rl-e', .95, 1064, agent_statistics={'no_delegation': 1}),
            result('depth-f', 1, 1065)]  # Missing telemetry is not a zero-delegation episode.
    _, values = timeline(rows, manifest([r['task_id'] for r in rows]), .9, .5)
    assert values['Successful episodes without delegation', 'Count'] == 1
    assert values['Subagent telemetry successful episodes', 'Delegation calls'] == 2


def test_subagent_means_only_include_successful_traces_and_use_separate_coverage():
    rows = [result('depth-0', 1, 1060, agent_statistics={'created': 4, 'peak_concurrency': 3}),
            result('depth-1', None, 1061, status='error', evaluation={'status': 'evaluation_error'},
                   agent_statistics={'created': 2}),
            result('depth-2', 0, 1062, agent_statistics={'created': 0, 'peak_concurrency': 0}),
            result('width-3', .9, 1063, agent_statistics={'created': 100}),
            result('width-4', .91, 1064, agent_statistics={'created': 2}),
            result('deepdive-qa_rl-5', .95, 1065, agent_statistics={'created': 100}),
            result('deepdive-qa_sft-6', 1, 1066, agent_statistics={'created': 0, 'peak_concurrency': 0}),
            result('depth-7', 1, 1067, cleanup_errors=['missing worker'], agent_statistics={'created': 100})]
    _, values = timeline(rows, manifest(['depth-0', 'depth-1', 'depth-2']), .9, .5)
    assert values['Subagents per successful episode', 'Successful new calls'] == 2
    assert values['Subagents per successful episode', 'Peak concurrency (includes queued runs)'] == 1.5
    assert values['Subagent telemetry successful episodes', 'Successful new calls'] == 3
    assert values['Subagent telemetry successful episodes', 'Peak concurrency (includes queued runs)'] == 2
