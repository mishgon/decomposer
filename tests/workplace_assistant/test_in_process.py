import asyncio
import json

from gyms.workplace_assistant.in_process import Workplace


def test_native_tools_are_isolated_and_execution_log_drives_score():
    first, second = Workplace(), Workplace()
    args = {'time_min': '2023-11-23', 'time_max': '2023-11-30',
            'value_to_plot': 'total_visits', 'plot_type': 'line'}
    reference = [{'name': 'analytics_create_plot', 'arguments': json.dumps(args)}]
    tool = next(t for t in first.tools() if t.name == 'analytics_create_plot')
    assert not first.score(reference)
    asyncio.run(tool.ainvoke(args))
    assert first.score(reference)
    assert not second.score(reference)
    assert first.actions == reference


def test_screen_rejects_overlap_and_noop_tasks():
    from evals.workplace_assistant.candidates import screen, SYSTEM
    def row(id, text, ground_truth):
        return {'id': id, 'responses_create_params': {'input': [
            {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}]},
            'ground_truth': ground_truth, 'category': 'analytics', 'environment_name': 'workplace_assistant'}
    action = [{'name': 'analytics_create_plot', 'arguments': json.dumps({
        'time_min': '2023-11-23', 'time_max': '2023-11-30',
        'value_to_plot': 'total_visits', 'plot_type': 'line'})}]
    accepted, rejected = screen([row(0, 'Seen request', action)], [
        row(1, ' SEEN   request ', action), row(2, 'Do nothing', []), row(3, 'Make a plot', action)])
    assert [r['id'] for r in accepted] == [3]
    assert rejected == {'overlap_or_duplicate': 1, 'no_action_passes': 1}
