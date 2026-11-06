"""Verify the fixed worker used to calibrate Workplace-20 before allocating training."""
import asyncio

from decomposer.models import create_model

WORKER_MODEL = 'lmrouter/qwen_3_5_4b_unlooped_non_thinking'


async def check():
    worker = create_model(WORKER_MODEL)
    try:
        await asyncio.wait_for(worker.ainvoke('Reply OK.', max_tokens=16), 30)
    except Exception as exc:
        raise RuntimeError(f'Workplace validation worker preflight failed ({WORKER_MODEL}): '
                           f'{type(exc).__name__}. Verify the calibrated worker before launching.') from exc
    finally:
        worker.http_client.close()
        await worker.http_async_client.aclose()
