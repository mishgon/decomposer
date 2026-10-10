import asyncio
import time

from langgraph_sdk import get_client


async def _cancel_agent_runs(client):
    """Stop all runs on a dedicated server, including uncheckpointed runs."""
    stopped = []
    offset = 0
    while True:
        threads = await client.threads.search(limit=100, offset=offset)
        for thread in threads:
            thread_id = thread["thread_id"]
            for status in ("pending", "running"):
                # Cancellation removes a run from this query; keep offset zero.
                while runs := await client.runs.list(thread_id, status=status, limit=100):
                    for run in runs:
                        await client.runs.cancel(thread_id, run["run_id"], wait=True, action="interrupt")
                        final = await client.runs.get(thread_id, run["run_id"])
                        if final["status"] in {"pending", "running"}:
                            raise RuntimeError(f"Agent run still active: {run['run_id']}")
                        stopped.append({"thread_id": thread_id, "run_id": run["run_id"],
                                        "status": final["status"]})
        if len(threads) < 100:
            return stopped
        offset += len(threads)


async def invoke_and_capture(
    url, assistant_id, inputs, *, thread_id=None, timeout=None, config=None,
):
    """Run on a dedicated server; stop its runs and return the trace and any error.

    Terminal statuses are added to the exported state, without editing checkpoints.
    Callers must save the trace before re-raising the returned error.
    """
    state = {}
    agent_error = None
    async with get_client(url=url, timeout=60) as client:
        thread = await client.threads.create(thread_id=thread_id)
        thread_id = thread["thread_id"]
        try:
            state = await asyncio.wait_for(client.runs.wait(
                thread_id, assistant_id, input=inputs, config=config,
            ), timeout)
            if "__error__" in state:
                agent_error = RuntimeError(str(state["__error__"]))
        except BaseException as error:
            agent_error = error

        try:
            shutdown = await asyncio.wait_for(_cancel_agent_runs(client), 60)
        except BaseException as error:
            shutdown_error = repr(error)
            agent_error = agent_error or error
        else:
            shutdown_error = None

        try:
            state = (await client.threads.get_state(thread_id))["values"]
            for run in state.get("decomposer_agent_runs", []):
                if "collected_at" in run:
                    continue
                final = await client.runs.get(thread_id, run["run_id"])
                if final["status"] in {"pending", "running"}:
                    continue
                run["status"] = "responded" if final["status"] == "success" else final["status"]
                run["collected_at"] = time.time()
                if run["status"] != "responded":
                    agent_error = agent_error or RuntimeError(f"Agent run {run['status']}")
                    run["error"] = repr(agent_error)
        except BaseException as error:
            state["agent_capture_error"] = repr(error)
            agent_error = agent_error or error
        if shutdown_error:
            state["agent_shutdown_error"] = shutdown_error
        else:
            state["agent_shutdown"] = shutdown
            stopped = {run["run_id"]: run["status"] for run in shutdown}
            for run in state.get("agent_runs", {}).values():
                if run["run_id"] in stopped and "collected_at" not in run:
                    status = stopped[run["run_id"]]
                    run["status"] = "responded" if status == "success" else status
        state["agent_error"] = repr(agent_error) if agent_error is not None else None
    return state, agent_error
