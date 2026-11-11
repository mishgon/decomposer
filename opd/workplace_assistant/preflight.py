"""Verify the fixed worker used to calibrate Workplace-20 before allocating training."""
from decomposer.models import create_model

WORKER_MODEL = 'lmrouter/qwen_3_5_4b_non_thinking'


async def check():
    worker = create_model(WORKER_MODEL)
    try:
        # The lmrouter transport owns request timeouts and outage recovery.
        # An outer deadline would cancel its retries and abort the whole job.
        await worker.ainvoke('Reply OK.', max_tokens=16)
    except Exception as exc:
        raise RuntimeError(f'Workplace validation worker preflight failed ({WORKER_MODEL}): '
                           f'{type(exc).__name__}. Verify the calibrated worker before launching.') from exc
    finally:
        worker.http_client.close()
        await worker.http_async_client.aclose()
