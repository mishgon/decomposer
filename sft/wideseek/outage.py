"""Pause collection on transient service outages without spending task attempts."""
import asyncio
import json
import random
import re
import time

from gyms.wideseek import agents
from gyms.wideseek.runtime import close_model, model, save


_TRANSIENT = re.compile(
    r'\b(APIConnectionError|APITimeoutError|ConnectError|ConnectTimeout|ReadError|'
    r'ReadTimeout|RemoteProtocolError|RateLimitError|InternalServerError)\b|'
    r'\b(?:HTTPStatusError: |Error code: |HTTP(?:/\S+)?[ :]+|status code[ :]+|'
    r'''(?:Client|Server) error ['"])(?:408|429|500|502|503|504)\b''',
    re.IGNORECASE,
)
_FATAL = re.compile(r'\b(AuthenticationError|PermissionDeniedError|NotFoundError|BadRequestError)\b|'
                    r'''\b(?:HTTPStatusError: |Error code: |Client error ['"])(?:400|401|403|404)\b''')


def transient(error):
    return bool(_TRANSIENT.search(str(error))) and not _FATAL.search(str(error))


def execution_errors(result, directory):
    errors = [result.get('error', '')]
    evaluation = result.get('evaluation', {})
    errors.extend(evaluation.get('errors', []))
    errors.append(evaluation.get('error', ''))
    log = directory / 'model_calls.jsonl'
    if log.exists():
        errors.extend(json.loads(line).get('error', '') for line in log.read_text().splitlines())
    return [error for error in errors if error]


async def probe_models(agent):
    # Exercise authenticated generation, not just HTTP health or model listing.
    for profile in dict.fromkeys((agents.AGENT_MODELS[agent], agents.RESEARCHER_MODEL, agents.JUDGE_MODEL)):
        policy = model(profile)
        try:
            async with asyncio.timeout(60):
                await policy.ainvoke('Reply OK.', max_tokens=1024)
        finally:
            await close_model(policy)


class OutageGuard:
    def __init__(self, root, probe):
        self.path = root / 'outage.json'
        self.probe = probe
        self.lock = asyncio.Lock()
        self.since = None
        self.failures = 0
        self.reason = None

    def pause(self, reason):
        if self.since is not None:
            return
        self.since = time.time()
        self.reason = reason
        self.write('paused')

    def write(self, status, **extra):
        save(self.path, dict(status=status, since=self.since, failures=self.failures,
                             reason=self.reason, updated_at=time.time(), **extra))

    async def wait(self):
        async with self.lock:
            while self.since is not None:
                ceiling = min(300, 5 * 2 ** min(self.failures, 6))
                delay = random.uniform(ceiling / 2, ceiling)
                self.write('paused', next_retry_at=time.time() + delay)
                await asyncio.sleep(delay)
                try:
                    await self.probe()
                except Exception as exc:
                    if not isinstance(exc, TimeoutError) and not transient(f'{type(exc).__name__}: {exc}'):
                        self.write('failed', reason_detail=type(exc).__name__)
                        raise
                    self.failures += 1
                    self.reason = type(exc).__name__
                else:
                    self.write('recovered', recovered_at=time.time())
                    self.since = None
                    self.failures = 0


async def collect_episode(guard, run_episode, task, mode, attempt, root, args, **kwargs):
    result_path = root / mode / task['task_id'] / f'attempt-{attempt:03d}' / 'result.json'
    while True:
        await guard.wait()
        await run_episode(task, mode, attempt, root, args, **kwargs)
        result = json.loads(result_path.read_text())
        directory = result_path.parent / result['execution_directory']
        if any(result.get(key) for key in ('cleanup_errors', 'capture_error', 'trace_error', 'export_error')):
            raise RuntimeError(f"Unsafe episode cleanup/export: {result_path}")
        errors = execution_errors(result, directory)
        if any(_FATAL.search(error) for error in errors):
            raise RuntimeError(f"Permanent service error: {result_path}")
        retry = any(transient(error) for error in errors)
        if not retry:
            if result.get('status') == 'error' and not str(result.get('error', '')).startswith('GraphRecursionError:'):
                raise RuntimeError(f"Unclassified episode error: {result_path}")
            if result.get('evaluation', {}).get('status') in {'judge_error', 'evaluation_error'}:
                raise RuntimeError(f"Invalid evaluation: {result_path}")
            return result
        # Keep the complete raw execution, but remove it from completed-attempt indexing.
        result['status'] = 'infrastructure_error'
        result['original_evaluation'] = result.get('evaluation')
        result['evaluation'] = {'status': 'infrastructure_error', 'score': None}
        save(result_path, result)
        result_path.replace(directory / 'infrastructure-result.json')
        guard.pause('Transient model/service request failure')
