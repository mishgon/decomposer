"""Stop outstanding subagents before evaluating their environment."""

import asyncio
import httpx


async def cancel_subagents(client, runs, subagents):
    for run in runs.values():
        if "response_sequence_number" in run:
            continue
        thread_id = subagents[run["subagent_id"]]["thread_id"]
        try:
            await asyncio.wait_for(client.runs.cancel(thread_id, run["run_id"], wait=True), 60)
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 404:
                raise
            # Completed runs may already have been removed. A 404 alone is not
            # sufficient: verify that the thread has no live runs before scoring.
            for status in ("pending", "running"):
                remaining = await asyncio.wait_for(client.runs.list(thread_id, status=status, limit=1), 30)
                if remaining:
                    raise
