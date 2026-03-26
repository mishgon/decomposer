from __future__ import annotations

import multiprocessing as mp
import time

from sympy import sympify

from .pattern_labels import label_trace_pattern
from .runtime_profiler import RuntimeProfiler
from .schemas import Episode
from .serializers import serialize_value
from .sympy_wrappers import ENTRYPOINTS


def _run_in_subprocess(entrypoint_name: str, kwargs: dict, max_calls: int, max_depth: int, queue) -> None:
    entrypoint = ENTRYPOINTS[entrypoint_name]
    profiler = RuntimeProfiler(entrypoint_name=entrypoint_name, max_calls=max_calls, max_depth=max_depth)
    started = time.perf_counter()
    try:
        prepared_kwargs = _prepare_kwargs(kwargs)
        result, calls, roots, truncated = profiler.run(entrypoint, **prepared_kwargs)
        payload = {
            "status": "trace_truncated" if truncated else "ok",
            "result": serialize_value(result),
            "calls": {call_id: node.to_dict() for call_id, node in calls.items()},
            "roots": roots,
            "duration_ms": (time.perf_counter() - started) * 1000,
        }
    except Exception as exc:
        payload = {
            "status": "exception",
            "result": serialize_value(exc),
            "calls": {},
            "roots": [],
            "duration_ms": (time.perf_counter() - started) * 1000,
        }
    queue.put(payload)


def _prepare_kwargs(kwargs: dict) -> dict:
    prepared = dict(kwargs)
    if "expr" in prepared:
        prepared["expr"] = sympify(prepared["expr"])
    return prepared


class EpisodeRunner:
    def __init__(self, *, timeout_s: float = 5.0, max_calls: int = 5000, max_depth: int = 50) -> None:
        self.timeout_s = timeout_s
        self.max_calls = max_calls
        self.max_depth = max_depth

    def run_episode(self, *, episode_id: str, entrypoint_name: str, kwargs: dict, input_seed: int = 0) -> dict:
        queue: mp.Queue = mp.Queue()
        process = mp.Process(
            target=_run_in_subprocess,
            args=(entrypoint_name, kwargs, self.max_calls, self.max_depth, queue),
        )
        started = time.perf_counter()
        process.start()
        process.join(self.timeout_s)
        if process.is_alive():
            process.terminate()
            process.join()
            episode = Episode(
                episode_id=episode_id,
                entry_func_id=entrypoint_name,
                task_family=entrypoint_name,
                input_seed=input_seed,
                input={k: serialize_value(v) for k, v in kwargs.items()},
                final_output=None,
                root_call_id=None,
                call_ids=[],
                status="timeout",
                duration_ms=(time.perf_counter() - started) * 1000,
                max_depth=0,
                num_calls=0,
                pattern_label="chain",
                metadata={"entrypoint_name": entrypoint_name},
            )
            return {"episode": episode.to_dict(), "calls": {}}

        payload = queue.get()
        calls = payload["calls"]
        root_call_id = payload["roots"][0] if payload["roots"] else None
        max_depth = max((node["depth"] for node in calls.values()), default=0)
        episode = Episode(
            episode_id=episode_id,
            entry_func_id=entrypoint_name,
            task_family=entrypoint_name,
            input_seed=input_seed,
            input={k: serialize_value(v) for k, v in kwargs.items()},
            final_output=payload["result"],
            root_call_id=root_call_id,
            call_ids=list(calls),
            status=payload["status"],
            duration_ms=payload["duration_ms"],
            max_depth=max_depth,
            num_calls=len(calls),
            pattern_label=label_trace_pattern_from_dicts(calls, root_call_id),
            metadata={"entrypoint_name": entrypoint_name},
        )
        return {"episode": episode.to_dict(), "calls": calls}


def label_trace_pattern_from_dicts(calls: dict[str, dict], root_call_id: str | None) -> str:
    from .schemas import CallNode, ExceptionInfo

    nodes = {}
    for call_id, raw in calls.items():
        exception = raw["exception"]
        nodes[call_id] = CallNode(
            call_id=call_id,
            func_id=raw["func_id"],
            parent_call_id=raw["parent_call_id"],
            depth=raw["depth"],
            inputs=raw["inputs"],
            output=raw["output"],
            exception=ExceptionInfo(**exception) if exception else None,
            subcalls=raw["subcalls"],
            start_ns=raw["start_ns"],
            end_ns=raw["end_ns"],
            thread_id=raw["thread_id"],
        )
    return label_trace_pattern(nodes, root_call_id)
