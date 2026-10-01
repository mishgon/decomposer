"""Small raw-trajectory statistics, shared by execution and monitoring."""
import json


def subagent_counts(messages):
    """Runs stay unawaited until wait returns their report, even if already done."""
    calls, spawned, active = {}, set(), set()
    peak = 0
    for message in messages:
        if message.get('type') == 'ai':
            calls.update({c['id']: c['name'] for c in message.get('tool_calls', [])})
        if message.get('type') != 'tool':
            continue
        name = calls.get(message.get('tool_call_id'))
        try:
            result = json.loads(message.get('content', ''))
        except (ValueError, TypeError):
            continue
        if name in {'run', 'spawn_subagent'} and isinstance(result, dict) and result.get('subagent_run_id'):
            worker = result['subagent_run_id']
            spawned.add(worker)
            active.add(worker)
            peak = max(peak, len(active))
        elif name == 'wait' and isinstance(result, list):
            for report in result:
                if isinstance(report, dict):
                    active.discard(report.get('subagent_run_id'))
    return len(spawned), peak
