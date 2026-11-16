"""Small raw-trajectory statistics, shared by execution and monitoring."""
import json
import math
from collections import Counter
from datetime import datetime


def subagent_bucket(statistics):
    """Bucket distinct agents with a successfully launched run, including forks."""
    return min(statistics['used'], 3) if 'used' in statistics else None


def episode_subagent_statistics(trace, archives):
    """Successful new calls, distinct used agents, and peak live worker runs.

    Worker creation/terminal timestamps avoid counting finished but unawaited
    runs as concurrent. Missing archives mean unknown concurrency, not zero.
    """
    statistics = {}
    if 'messages' in trace:
        calls, created, targets, used = {}, set(), {}, set()
        missing_target = False
        delegated = False
        for message in trace['messages']:
            message = message.get('data', message)
            if message.get('type') == 'ai':
                calls.update({c['id']: c['name'] for c in message.get('tool_calls', [])})
                targets.update({c['id']: c.get('args', {}).get('agent_id')
                                for c in message.get('tool_calls', []) if c['name'] == 'run'})
                delegated |= any(c['name'] in {'new', 'fork', 'run'}
                                 for c in message.get('tool_calls', []))
            call_id = message.get('tool_call_id')
            if message.get('type') != 'tool' or calls.get(call_id) not in {'new', 'run'}:
                continue
            try:
                response = json.loads(message.get('content', ''))
            except (ValueError, TypeError):
                continue
            if not isinstance(response, dict) or response.get('error'):
                continue
            if calls[call_id] == 'new' and response.get('agent_id'):
                created.add(call_id)
            if calls[call_id] == 'run' and response.get('agent_run_id'):
                agent = targets.get(call_id) or trace.get('agent_runs', {}).get(
                    response['agent_run_id'], {}).get('agent_id')
                if agent is None:
                    missing_target = True
                else:
                    used.add(agent)
        used.update(run['agent_id'] for run in trace.get('agent_runs', {}).values())
        statistics['created'] = len(created)
        statistics['no_delegation'] = int(not delegated)
        if not missing_target:
            statistics['used'] = len(used)
    if 'agents' not in trace or 'agent_runs' not in trace:
        return statistics
    threads = {agent['thread_id'] for agent in trace['agents'].values()}
    if not threads.issubset(archives):
        return statistics
    events, captured = [], set()
    for thread_id in threads:
        if 'runs' not in archives[thread_id]:
            return statistics
        for run in archives[thread_id]['runs']:
            if run['run_id'] in captured:
                continue
            if run['status'] not in {'success', 'error', 'interrupted', 'timeout'}:
                return statistics
            try:
                start = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00')).timestamp()
                end = datetime.fromisoformat(run['updated_at'].replace('Z', '+00:00')).timestamp()
            except (KeyError, ValueError, TypeError):
                return statistics
            if not (math.isfinite(start) and math.isfinite(end) and end >= start):
                return statistics
            captured.add(run['run_id'])
            if end > start:
                events.extend(((start, 1, thread_id), (end, -1, thread_id)))
    if not {r['run_id'] for r in trace['agent_runs'].values()}.issubset(captured):
        return statistics
    active, peak = Counter(), 0
    for _, delta, thread_id in sorted(events):
        active[thread_id] += delta
        if not active[thread_id]:
            del active[thread_id]
        peak = max(peak, len(active))
    statistics['peak_concurrency'] = peak
    return statistics


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
        if name == 'run' and isinstance(result, dict) and result.get('agent_run_id'):
            worker = result['agent_run_id']
            spawned.add(worker)
            active.add(worker)
            peak = max(peak, len(active))
        elif name == 'wait' and isinstance(result, list):
            for report in result:
                if isinstance(report, dict):
                    active.discard(report.get('agent_run_id'))
    return len(spawned), peak
