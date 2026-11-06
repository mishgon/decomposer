"""Retry transport-failed episodes in fresh workspaces, never replay a tool POST."""
import asyncio


class EpisodeTransportError(RuntimeError):
    pass


async def retry_episode(run):
    for attempt in range(3):
        try:
            return await run()
        except EpisodeTransportError:
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)
