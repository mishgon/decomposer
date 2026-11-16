import pytest

from evals.wideseek.clearml_reporter import snapshot, timeline
from gyms.wideseek.runtime import save


def result(task, score, end, **extra):
    return {'task_id': task, 'started_at': end - 60, 'finished_at': end,
            'status': 'finished', 'evaluation': {'status': 'scored', 'score': score}, **extra}


def manifest(tasks):
    return {'started_at': 1000, 'settings': {'tasks': tasks, 'repetitions': 4, 'modes': ['decomposer']}}


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
