"""Small raw-trajectory statistics, shared by execution and monitoring."""
import json
import math


def qualifies(result, threshold):
    """Whether a finished, captured episode meets a native-score threshold."""
    score = result["evaluation"].get("score")
    return (result["status"] == "finished" and not result.get("cleanup_errors")
            and result.get("trace_available", True)
            and result["evaluation"].get("status") == "scored"
            and type(score) in (int, float) and math.isfinite(score)
            and 0 <= score <= 1 and (score == 1. or score > threshold))


def subagent_counts(messages):
    """Runs stay unawaited until wait returns their report, even if already done."""
    calls, spawned, active = {}, set(), set()
    peak = 0
    for message in messages:
        message = message.get('data', message)
        if message.get('type') == 'ai':
            calls.update({c['id']: c['name'] for c in message.get('tool_calls', [])})
        if message.get('type') != 'tool':
            continue
        name = calls.get(message.get('tool_call_id'))
        try:
            result = json.loads(message.get('content', ''))
        except (ValueError, TypeError):
            continue
        if name in {'run', 'spawn_subagent'} and isinstance(result, dict) and (result.get('agent_run_id') or result.get('subagent_run_id')):
            worker = result.get('agent_run_id') or result['subagent_run_id']
            spawned.add(worker)
            active.add(worker)
            peak = max(peak, len(active))
        elif name == 'wait' and isinstance(result, list):
            for report in result:
                if isinstance(report, dict):
                    active.discard(report.get('agent_run_id') or report.get('subagent_run_id'))
    return len(spawned), peak
