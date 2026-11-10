"""Retry interrupted episodes in fresh workspaces; preserve independent results."""
import asyncio
import logging
import random
import re

from opd.policy import is_episode_failure


log = logging.getLogger(__name__)
_TRANSIENT = re.compile(
    r'\b(APIConnectionError|APITimeoutError|ConnectError|ConnectTimeout|ReadError|'
    r'WriteError|ReadTimeout|WriteTimeout|RemoteProtocolError|RateLimitError|InternalServerError)\b|'
    r'\b(?:Error code: |HTTP(?:/\S+)?[ :]+|status code[ :]+)(?:408|429|500|502|503|504)\b|'
    r"^RuntimeError(?:: Policy inference request timed out|\('Policy inference request timed out'\))$")


class EpisodeTransportError(RuntimeError):
    pass


def check_episode(result, *, has_tokens):
    """Recognize serialized transient failures after the caller saves the raw result."""
    if any(result.get(key) for key in ('cleanup_errors', 'capture_error', 'trace_error', 'export_error')):
        raise RuntimeError('Episode cleanup/capture/export failed; see saved result')
    errors = list(result.get('worker_errors', []))
    error = result.get('error')
    if error and not (errors and str(error).startswith('WorkerError:')):
        errors.append(error)
    evaluation = result.get('evaluation', {})
    errors.extend(evaluation.get('errors', []))
    if evaluation.get('error'):
        errors.append(evaluation['error'])
    transient = False
    unexpected = False
    for error in errors:
        if _TRANSIENT.search(str(error)):
            transient = True
        elif not is_episode_failure({'error_type': str(error).split(':', 1)[0], 'error': str(error)}):
            # WideSeek's timeout message has no exception type.
            if not (result.get('status') == 'timeout' and error == result.get('error')):
                unexpected = True
    if transient and unexpected:
        raise RuntimeError('Mixed transient and non-transient episode failure; see saved result')
    if transient:
        raise EpisodeTransportError('Transient worker/judge failure; raw episode retained')
    if not has_tokens and (result.get('status') == 'timeout' or result.get('error_type') == 'TimeoutError'):
        raise EpisodeTransportError('Episode deadline expired before student tokens; raw episode retained')


async def retry_episode(run):
    attempt = 0
    while True:
        try:
            return await run()
        except EpisodeTransportError:
            ceiling = min(300, 5 * 2 ** min(attempt, 6))
            delay = random.uniform(ceiling / 2, ceiling)
            log.warning('OPD episode interrupted; retry %d in %.1fs; completed episodes retained', attempt + 1, delay)
            await asyncio.sleep(delay)
            attempt += 1
