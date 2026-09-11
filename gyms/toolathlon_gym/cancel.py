"""Stop outstanding subagents before evaluating their environment."""

import asyncio
import httpx


async def cancel_subagents(client, runs):
    for run in runs.values():
        if run.get("report") is not None:
            continue
        try:
            await asyncio.wait_for(client.runs.cancel(run["thread_id"], run["run_id"], wait=True), 60)
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 404:
                raise
            # Completed runs may already have been removed. A 404 alone is not
            # sufficient: verify that the thread has no live runs before scoring.
            for status in ("pending", "running"):
                remaining = await asyncio.wait_for(client.runs.list(run["thread_id"], status=status, limit=1), 30)
                if remaining:
                    raise
