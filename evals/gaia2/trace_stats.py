"""Per-rollout cost statistics for completed GAIA2 runs.

The accuracy tables say how well each system scores; this module says what it
spent. It reads only finished artifacts and never writes into a results
directory.

Three sources are joined per rollout:

* ``decomposer_sidecars/<scenario>__run<k>.json`` — the manager's own message
  trace (with exact ``usage_metadata`` on every AI message), the spawn/wait
  sequence, each subagent's prompt/report/tool-calls, the per-episode tool
  schemas, and every environment tool call with a wall-clock ``started_at``.
* ``logs/langgraph_subagent.log`` — one line per subagent run carrying
  ``run_id``/``run_started_at``/``run_ended_at``/``run_wait_time_ms``.
  ``spawn_subagent`` returns the LangGraph ``run_id`` as ``subagent_run_id``
  (``src/decomposer/core.py``), so the join is exact.
* ``logs/{manager,worker,policy,manager_worker}_vllm.log`` — periodic throughput
  samples, integrated to recover token volumes the traces do not record.

Parallelism is measured twice on purpose. The manager cannot emit parallel tool
calls, so concurrency exists only when it issues several ``spawn_subagent``
calls before a ``wait``; the *structural* measure reads that intent off the
message sequence. Whether the subagents then actually overlapped is a different
question, answered by the *temporal* measure over their real intervals.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

# Defaults for Hertz-2. The DeepSeek teacher runs live on occ-cds under a
# different mount, so both roots are overridable from the command line.
RESULTS_ROOT = Path(
    "/home/sukhorukov/decomposer_artifacts/evaluation/gaia2/results/validation"
)
ANALYSIS_ROOT = Path(
    "/home/sukhorukov/decomposer_artifacts/analysis/gaia2/trace_stats"
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_FIELD = re.compile(r"(\w+)=(\S+)")
_VLLM = re.compile(
    r"INFO (\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}).*?"
    r"Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s"
)
_PREFIX_HIT = re.compile(r"Prefix cache hit rate: ([\d.]+)")

# Dividing the prefilled total by (1 - hit) recovers the logical context, but the
# factor is 1/(1-hit): at 0.98 it is 50x and at 0.998 it is 500x, so a rounding
# error in the last reported digit swings the answer by an order of magnitude.
# Above this the estimate is discarded rather than reported with false precision.
MAX_TRUSTED_HIT_RATE = 0.98

# `wait` returns this string rather than a report list when nothing is running.
_NO_RUNNING = "No running subagents"


# --------------------------------------------------------------------------
# log parsing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Lifecycle:
    """One subagent run as the LangGraph server saw it."""

    run_id: str
    status: str
    created: float | None
    started: float | None
    ended: float | None
    exec_seconds: float | None
    wait_seconds: float | None


def _iso(value: str | None) -> float | None:
    if not value or value == "None":
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _ms(value: str | None) -> float | None:
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return None


def parse_langgraph_log(path: Path) -> dict[str, Lifecycle]:
    """Map ``run_id`` to its lifecycle for every terminated subagent run."""
    out: dict[str, Lifecycle] = {}
    if not path.exists():
        return out
    with path.open(errors="ignore") as handle:
        for line in handle:
            if "Background run " not in line:
                continue
            clean = _ANSI.sub("", line)
            fields = dict(_FIELD.findall(clean))
            run_id = fields.get("run_id")
            if not run_id:
                continue
            out[run_id] = Lifecycle(
                run_id=run_id,
                status=clean.split("Background run ", 1)[1].split()[0],
                created=_iso(fields.get("run_created_at")),
                started=_iso(fields.get("run_started_at")),
                ended=_iso(fields.get("run_ended_at")),
                exec_seconds=_ms(fields.get("run_exec_ms")),
                wait_seconds=_ms(fields.get("run_wait_time_ms")),
            )
    return out


@dataclass
class Throughput:
    """Token volume recovered by integrating a vLLM server's throughput log."""

    prompt_tokens: float = 0.0
    generation_tokens: float = 0.0
    seconds: float = 0.0
    samples: int = 0
    prefix_hit_rate: float | None = None

    @property
    def logical_prompt_tokens(self) -> float | None:
        """Prompt tokens before the prefix cache absorbed the repeated prefix.

        The throughput counter sees only what was actually prefilled, so at a
        94 % hit rate it under-reports the context the model was served by more
        than an order of magnitude. Dividing back out recovers the logical
        figure a token bill would show.
        """
        if self.prefix_hit_rate is None or self.prefix_hit_rate > MAX_TRUSTED_HIT_RATE:
            return None
        return self.prompt_tokens / (1.0 - self.prefix_hit_rate)

    def as_dict(self) -> dict[str, Any]:
        logical = self.logical_prompt_tokens
        return {
            "prompt_tokens_computed": round(self.prompt_tokens),
            "prompt_tokens_logical_implied": round(logical) if logical else None,
            "prefix_hit_rate": self.prefix_hit_rate,
            "generation_tokens": round(self.generation_tokens),
            "seconds": round(self.seconds, 1),
            "samples": self.samples,
        }


def parse_vllm_throughput(path: Path, year: int) -> Throughput:
    """Integrate ``Avg * throughput`` samples into a token count.

    Each sample reports the mean rate since the previous sample, so the interval
    is the gap to the *preceding* line. vLLM stamps month-day-time without a
    year; ``year`` seeds it and a backwards step is treated as a rollover.
    """
    total = Throughput()
    if not path.exists():
        return total
    previous: datetime | None = None
    with path.open(errors="ignore") as handle:
        for line in handle:
            match = _VLLM.search(line)
            if not match:
                continue
            month, day, hour, minute, second = (int(x) for x in match.groups()[:5])
            stamp = datetime(year, month, day, hour, minute, second)
            if previous is not None and stamp < previous:
                stamp += timedelta(days=365)
            prompt_rate = float(match.group(6))
            generation_rate = float(match.group(7))
            if previous is not None:
                delta = (stamp - previous).total_seconds()
                # A long gap means the server was idle or restarted; charging a
                # stale rate across it would invent tokens.
                if 0 < delta <= 120:
                    total.prompt_tokens += prompt_rate * delta
                    total.generation_tokens += generation_rate * delta
                    total.seconds += delta
            total.samples += 1
            previous = stamp
            hit = _PREFIX_HIT.search(line)
            if hit:
                # Cumulative over the run, so the last sample is the run figure.
                total.prefix_hit_rate = float(hit.group(1)) / 100.0
    return total


# --------------------------------------------------------------------------
# parallelism
# --------------------------------------------------------------------------


def _tool_name(message: dict[str, Any]) -> str | None:
    calls = (message.get("data") or {}).get("tool_calls") or []
    return calls[0].get("name") if calls else None


def structural_parallelism(messages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Replay the manager's spawn/wait sequence and track the outstanding set.

    Returns the depth reached after each spawn (how many subagents the manager
    deliberately had in flight), how many reports each ``wait`` collected, and
    the length of each unbroken run of spawns.
    """
    outstanding: set[str] = set()
    depths: list[int] = []
    per_wait: list[int] = []
    bursts: list[int] = []
    burst = 0
    pending: str | None = None
    spawn_errors = 0
    empty_waits = 0

    for message in messages:
        kind = message.get("type")
        if kind == "ai":
            pending = _tool_name(message)
            continue
        if kind != "tool" or pending is None:
            continue
        content = (message.get("data") or {}).get("content")
        payload: Any = None
        if isinstance(content, str):
            try:
                payload = json.loads(content)
            except (ValueError, TypeError):
                payload = None

        if pending == "spawn_subagent":
            if isinstance(payload, dict) and "subagent_run_id" in payload:
                outstanding.add(payload["subagent_run_id"])
                depths.append(len(outstanding))
                burst += 1
            else:
                spawn_errors += 1
        elif pending == "wait":
            if burst:
                bursts.append(burst)
                burst = 0
            if isinstance(payload, list):
                per_wait.append(len(payload))
                for report in payload:
                    if isinstance(report, dict):
                        outstanding.discard(report.get("subagent_run_id"))
            else:
                empty_waits += 1
        pending = None

    if burst:
        bursts.append(burst)

    return {
        "max_outstanding": max(depths, default=0),
        "mean_outstanding_at_spawn": round(_mean(depths), 3),
        "waits": len(per_wait),
        "max_reports_per_wait": max(per_wait, default=0),
        "mean_reports_per_wait": round(_mean(per_wait), 3),
        "spawn_bursts": bursts,
        "max_burst": max(bursts, default=0),
        "spawn_errors": spawn_errors,
        "empty_waits": empty_waits,
        "unclosed_at_end": len(outstanding),
    }


def temporal_parallelism(
    intervals: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    """Sweep the real subagent intervals for overlap.

    ``mean_concurrent`` is time-weighted over the span in which any subagent was
    running, so it is not diluted by the manager's own thinking time;
    ``busy_sum / busy_union`` is the speedup delegation actually bought.
    """
    clean = [(lo, hi) for lo, hi in intervals if lo is not None and hi is not None and hi >= lo]
    if not clean:
        return {
            "max_concurrent": 0,
            "mean_concurrent": 0.0,
            "busy_union_seconds": 0.0,
            "busy_sum_seconds": 0.0,
            "overlap_factor": 0.0,
            "measured_subagents": 0,
        }

    events: list[tuple[float, int]] = []
    for lo, hi in clean:
        events.append((lo, 1))
        events.append((hi, -1))
    events.sort()

    active = 0
    peak = 0
    union = 0.0
    weighted = 0.0
    previous = events[0][0]
    for stamp, delta in events:
        if active > 0:
            span = stamp - previous
            union += span
            weighted += span * active
        previous = stamp
        active += delta
        peak = max(peak, active)

    busy_sum = sum(hi - lo for lo, hi in clean)
    return {
        "max_concurrent": peak,
        "mean_concurrent": round(weighted / union, 3) if union else 0.0,
        "busy_union_seconds": round(union, 3),
        "busy_sum_seconds": round(busy_sum, 3),
        "overlap_factor": round(busy_sum / union, 3) if union else 0.0,
        "measured_subagents": len(clean),
    }


def _merge_structural(per_turn: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fold per-turn spawn/wait statistics into one rollout-level record."""
    if not per_turn:
        return structural_parallelism([])
    bursts = [b for turn in per_turn for b in turn["spawn_bursts"]]
    return {
        "max_outstanding": max(t["max_outstanding"] for t in per_turn),
        "mean_outstanding_at_spawn": round(
            _mean([t["mean_outstanding_at_spawn"] for t in per_turn if t["waits"] or t["max_outstanding"]]),
            3,
        ),
        "waits": sum(t["waits"] for t in per_turn),
        "max_reports_per_wait": max(t["max_reports_per_wait"] for t in per_turn),
        "mean_reports_per_wait": round(
            _mean([t["mean_reports_per_wait"] for t in per_turn if t["waits"]]), 3
        ),
        "spawn_bursts": bursts,
        "max_burst": max(bursts, default=0),
        "spawn_errors": sum(t["spawn_errors"] for t in per_turn),
        "empty_waits": sum(t["empty_waits"] for t in per_turn),
        "unclosed_at_end": sum(t["unclosed_at_end"] for t in per_turn),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------
# tokenization
# --------------------------------------------------------------------------


class TokenCounter:
    """Token counter with memoisation, because tool results repeat verbatim."""

    def __init__(self, model_path: str | None):
        self._model_path = model_path
        self._tokenizer = None
        self._cache: dict[str, int] = {}

    def _load(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path, trust_remote_code=True
            )
        return self._tokenizer

    def count(self, text: Any) -> int:
        if text is None:
            return 0
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False, default=str)
        if not text:
            return 0
        key = text if len(text) < 512 else f"{len(text)}:{hash(text)}"
        hit = self._cache.get(key)
        if hit is None:
            hit = len(self._load().encode(text, add_special_tokens=False))
            self._cache[key] = hit
        return hit


# --------------------------------------------------------------------------
# per-rollout statistics
# --------------------------------------------------------------------------


@dataclass
class TokenModel:
    """Subagent tokens split into the part we know and the part we scale.

    The sidecar records a subagent's prompt, its tool calls and their results,
    but not the assistant text it wrote between them. That text enters the
    context of every later call, so input tokens are affine in the mean
    assistant length ``A``: ``base + a_coeff * A``. ``A`` is measured per run
    from the worker's generation throughput, which is exact because generated
    tokens are never served from the prefix cache.
    """

    base_input: float = 0.0
    a_coeff: float = 0.0
    calls: int = 0

    def input_tokens(self, mean_assistant: float) -> float:
        return self.base_input + self.a_coeff * mean_assistant

    def output_tokens(self, mean_assistant: float) -> float:
        return self.calls * mean_assistant


def join_tool_calls(
    subagents: Sequence[dict[str, Any]],
    env_calls: Sequence[dict[str, Any]],
    lifecycles: dict[str, Lifecycle],
) -> tuple[dict[str, list[Any]], dict[str, int]]:
    """Attribute each subagent's tool calls to the env records that ran them.

    The per-subagent record keeps only ``(name, args)``; the results and
    timestamps live in one flat list. Matching on ``(name, args)`` alone is
    ambiguous roughly half the time because subagents repeat identical reads, so
    candidates are first restricted to the subagent's own LangGraph interval.
    """
    buckets: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(env_calls):
        key = (record["tool"], json.dumps(record.get("arguments"), sort_keys=True, default=str))
        buckets.setdefault(key, []).append(index)

    taken: set[int] = set()
    results: dict[str, list[Any]] = {}
    quality = {"unique": 0, "ambiguous": 0, "unmatched": 0}

    for state in subagents:
        run_id = state["subagent_run_id"]
        life = lifecycles.get(run_id)
        low = life.started if life and life.started else -math.inf
        high = life.ended if life and life.ended else math.inf
        picked: list[Any] = []
        for call in state.get("tool_calls") or []:
            key = (call["name"], json.dumps(call.get("args"), sort_keys=True, default=str))
            candidates = [i for i in buckets.get(key, []) if i not in taken]
            inside = [i for i in candidates if low <= env_calls[i]["started_at"] <= high]
            if len(inside) == 1:
                quality["unique"] += 1
                chosen = inside[0]
            elif inside:
                quality["ambiguous"] += 1
                chosen = inside[0]
            elif candidates:
                quality["ambiguous"] += 1
                chosen = candidates[0]
            else:
                quality["unmatched"] += 1
                picked.append(None)
                continue
            taken.add(chosen)
            picked.append(env_calls[chosen].get("result"))
        results[run_id] = picked
    return results, quality


def rollout_stats(
    sidecar: Path,
    lifecycles: dict[str, Lifecycle],
    tokens: TokenCounter,
    schema_cache: dict[str, int],
    system_prompt_tokens: int,
) -> dict[str, Any]:
    """Every per-rollout quantity, with the subagent tokens left as a model."""
    data = json.loads(sidecar.read_text())
    env_calls = data.get("tool_calls") or []

    manager_in = manager_out = manager_calls = 0
    manager_reasoning = manager_cached_in = 0
    failures: list[str] = []
    per_turn: list[dict[str, Any]] = []
    subagent_states: list[dict[str, Any]] = []
    schema_tokens = 0
    elapsed = 0.0

    for turn in data.get("turns") or []:
        manager = turn.get("manager") or {}
        # A turn the manager failed carries `failure` in place of its usage,
        # message trace and subagent list. A reasoning model that runs past its
        # output budget lands here, so the kind is worth keeping.
        failure = manager.get("failure")
        if failure:
            failures.append(str(failure.get("kind") or "unknown"))
        usage = manager.get("usage") or {}
        manager_in += int(usage.get("input_tokens") or 0)
        manager_out += int(usage.get("output_tokens") or 0)
        elapsed += float((manager.get("timing") or {}).get("elapsed_seconds") or 0.0)

        trace = manager.get("trace") or {}
        messages = trace.get("manager_messages") or []
        for message in messages:
            usage_metadata = (message.get("data") or {}).get("usage_metadata")
            if not usage_metadata:
                continue
            manager_calls += 1
            # Only a remote reasoning API fills these in; a local non-thinking
            # vLLM leaves them empty, so they stay zero for every other system.
            manager_reasoning += int(
                (usage_metadata.get("output_token_details") or {}).get("reasoning") or 0
            )
            manager_cached_in += int(
                (usage_metadata.get("input_token_details") or {}).get("cache_read") or 0
            )
        # Turns share the scenario's tool schemas; count them once.
        if not schema_tokens:
            schemas = (trace.get("runtime_context") or {}).get("tool_schemas") or []
            blob = json.dumps(schemas, ensure_ascii=False, default=str)
            key = f"{len(blob)}:{hash(blob)}"
            if key not in schema_cache:
                schema_cache[key] = tokens.count(blob)
            schema_tokens = schema_cache[key]
        per_turn.append(structural_parallelism(messages))
        subagent_states.extend(manager.get("subagent_states") or [])

    structural = _merge_structural(per_turn)
    results, join_quality = join_tool_calls(subagent_states, env_calls, lifecycles)

    intervals: list[tuple[float, float]] = []
    queue_seconds = 0.0
    statuses: Counter[str] = Counter()
    model = TokenModel()
    shared = system_prompt_tokens + schema_tokens

    for state in subagent_states:
        run_id = state["subagent_run_id"]
        statuses[state.get("status") or "unknown"] += 1
        life = lifecycles.get(run_id)
        if life:
            if life.started is not None and life.ended is not None:
                intervals.append((life.started, life.ended))
            if life.wait_seconds:
                queue_seconds += life.wait_seconds

        prompt_tokens = tokens.count(state.get("prompt"))
        result_tokens = [tokens.count(r) for r in results.get(run_id, [])]
        calls = len(result_tokens) + 1
        model.calls += calls
        model.base_input += calls * (shared + prompt_tokens)
        model.base_input += sum(
            (calls - 1 - index) * value for index, value in enumerate(result_tokens)
        )
        model.a_coeff += calls * (calls - 1) / 2

    temporal = temporal_parallelism(intervals)
    tool_seconds = sum(float(c.get("latency_seconds") or 0.0) for c in env_calls)

    return {
        "scenario_id": data.get("scenario_id"),
        "run_number": data.get("run_number"),
        "turns": len(per_turn),
        "manager_failures": failures,
        # The sidecar clock covers the decomposer episode only; the rollout's
        # wall time (ARE setup, notification waits, oracle validation) comes
        # from the lite trace and is attached by the caller.
        "episode_seconds": round(elapsed, 3),
        "elapsed_seconds": round(elapsed, 3),
        "manager_only_seconds": round(max(elapsed - temporal["busy_union_seconds"], 0.0), 3),
        "subagent_queue_seconds": round(queue_seconds, 3),
        "env_tool_seconds": round(tool_seconds, 3),
        "env_tool_calls": len(env_calls),
        "spawns": len(subagent_states),
        "subagent_statuses": dict(statuses),
        "manager_input_tokens": manager_in,
        "manager_output_tokens": manager_out,
        "manager_reasoning_tokens": manager_reasoning,
        "manager_cached_input_tokens": manager_cached_in,
        "manager_calls": manager_calls,
        "subagent_calls": model.calls,
        "subagent_base_input_tokens": round(model.base_input),
        "subagent_input_a_coeff": model.a_coeff,
        "schema_tokens": schema_tokens,
        "structural": structural,
        "temporal": temporal,
        "tool_call_join": join_quality,
    }


# --------------------------------------------------------------------------
# run-level analysis
# --------------------------------------------------------------------------


def _wall_seconds(status: dict[str, Any]) -> float:
    """How long the run took, preferring its own tally over the timestamps."""
    wall = float(status.get("total_seconds") or 0.0)
    if wall:
        return wall
    try:
        return (
            datetime.fromisoformat(status["finished_at"])
            - datetime.fromisoformat(status["started_at"])
        ).total_seconds()
    except (KeyError, ValueError):
        return 0.0


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)
    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    return {
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": round(at(0.5), 3),
        "p90": round(at(0.9), 3),
        "max": round(ordered[-1], 3),
    }


def _outcomes(run_dir: Path) -> dict[tuple[str, int], float]:
    """Per-rollout score keyed by (scenario, run number).

    ``output.jsonl`` rows are not in run order, so the repeat index has to come
    from ``metadata.run_number`` rather than from the position in the file;
    deriving it by counting silently mispairs scores with rollouts.
    """
    path = run_dir / "output.jsonl"
    if not path.exists():
        return {}
    out: dict[tuple[str, int], float] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("metadata") or {}
        scenario = str(meta.get("scenario_id") or row.get("task_id"))
        run_number = meta.get("run_number")
        if run_number is None:
            continue
        out[(scenario, int(run_number))] = float(row.get("score") or 0.0)
    return out


def _lite_durations(
    run_dir: Path, ceiling: float | None = None
) -> tuple[dict[tuple[str, int], float], int]:
    """Wall time per rollout as the benchmark measured it, plus a sentinel count.

    Summed over the run and divided by its concurrency this reconciles with the
    run's own elapsed time; the decomposer's episode clock does not, because it
    excludes scenario setup, notification waits and oracle validation.

    An abandoned scenario can carry a placeholder duration of 600000 s, which is
    a sentinel and not a measurement. No rollout can outlast the run that
    contains it, so anything above the run's own wall clock is clamped there and
    counted; left alone, a single such row moves a cell's mean by hours.
    """
    out: dict[tuple[str, int], float] = {}
    clamped = 0
    for path in (run_dir / "lite").glob("*.json"):
        run_number = _run_number(path.name)
        if run_number is None:
            continue
        data = json.loads(path.read_text())
        value = float(data.get("run_duration") or 0.0)
        if ceiling and value > ceiling:
            value = ceiling
            clamped += 1
        out[(str(data.get("scenario_id")), run_number)] = value
    return out, clamped


def _worker_model_path(sidecar: Path) -> str | None:
    return _manager_config(sidecar, "subagent").get("path")


def _manager_config(sidecar: Path, role: str = "manager") -> dict[str, Any]:
    data = json.loads(sidecar.read_text())
    config = (data.get("configuration") or {}).get("model_configuration") or {}
    return config.get(role) or {}


def analyse_decomposer_run(run_dir: Path, limit: int | None = None) -> dict[str, Any]:
    """Aggregate one decomposer run directory."""
    status = json.loads((run_dir / "run_status.json").read_text())
    year = datetime.fromisoformat(status["finished_at"]).year
    lifecycles = parse_langgraph_log(run_dir / "logs" / "langgraph_subagent.log")

    population = sorted((run_dir / "decomposer_sidecars").glob("*.json"))
    if not population:
        raise ValueError(f"no sidecars under {run_dir}")
    sidecars = population[:limit] if limit else population
    # The vLLM logs cover the whole run, the sidecars may be a sample; comparing
    # the two without this factor overstates tokens by the sampling ratio.
    sampled = len(sidecars) / len(population)

    tokens = TokenCounter(_worker_model_path(sidecars[0]))
    manager_backend = _manager_config(sidecars[0]).get("backend")
    system_prompt_tokens = tokens.count(_subagent_system_prompt())
    schema_cache: dict[str, int] = {}

    rollouts = [
        rollout_stats(path, lifecycles, tokens, schema_cache, system_prompt_tokens)
        for path in sidecars
    ]

    worker_log = run_dir / "logs" / "worker_vllm.log"
    shared_log = run_dir / "logs" / "manager_worker_vllm.log"
    manager_log = run_dir / "logs" / "manager_vllm.log"
    worker = parse_vllm_throughput(
        worker_log if worker_log.exists() else shared_log, year
    )
    manager_measured = parse_vllm_throughput(manager_log, year) if manager_log.exists() else None

    manager_output_total = sum(r["manager_output_tokens"] for r in rollouts)

    # The throughput counter also sees retried and aborted requests, so it runs
    # high. The manager is the one role whose true usage is recorded, so its
    # integral-to-exact ratio calibrates the same instrument on the worker.
    calibration = 1.0
    calibration_source = "none"
    if manager_measured and manager_measured.generation_tokens:
        predicted = manager_output_total / len(rollouts) * len(population)
        calibration = predicted / manager_measured.generation_tokens
        calibration_source = "manager_vllm_log"

    generation = worker.generation_tokens * calibration
    implied = worker.logical_prompt_tokens
    worker_prompt = (implied * calibration) if implied else None
    if not worker_log.exists() and shared_log.exists():
        # One server fed both roles; the manager's share is known exactly.
        generation = max(generation - manager_output_total / sampled, 0.0)
        if worker_prompt is not None:
            worker_prompt = max(
                worker_prompt
                - sum(r["manager_input_tokens"] for r in rollouts) / sampled,
                0.0,
            )

    subagent_output_each = generation / len(population)
    subagent_calls = sum(r["subagent_calls"] for r in rollouts) or 1
    mean_assistant = generation * sampled / subagent_calls

    reconstructed_total = sum(
        TokenModel(
            base_input=r["subagent_base_input_tokens"],
            a_coeff=r["subagent_input_a_coeff"],
            calls=r["subagent_calls"],
        ).input_tokens(mean_assistant)
        for r in rollouts
    )
    if worker_prompt is None:
        # No usable measurement: the reconstruction stands alone, and the
        # summary says so rather than implying a measured figure.
        input_source = "reconstruction"
        subagent_input_each = reconstructed_total / len(rollouts)
    else:
        input_source = "measured"
        subagent_input_each = worker_prompt / len(population)

    for record in rollouts:
        # Per-rollout attribution of a run-level measurement: share it out by the
        # rollout's own subagent work rather than pretending every rollout cost
        # the mean.
        weight = record["subagent_calls"] / (subagent_calls / len(rollouts)) if subagent_calls else 0.0
        floor = TokenModel(
            base_input=record["subagent_base_input_tokens"],
            a_coeff=record["subagent_input_a_coeff"],
            calls=record["subagent_calls"],
        )
        record["subagent_input_floor_tokens"] = round(floor.input_tokens(mean_assistant))
        record["subagent_input_tokens"] = round(subagent_input_each * weight)
        record["subagent_output_tokens"] = round(subagent_output_each * weight)
        record["total_tokens"] = (
            record["manager_input_tokens"]
            + record["manager_output_tokens"]
            + record["subagent_input_tokens"]
            + record["subagent_output_tokens"]
        )

    scores = _outcomes(run_dir)
    durations, clamped = _lite_durations(run_dir, ceiling=_wall_seconds(status))
    for record in rollouts:
        key = (record["scenario_id"], record["run_number"])
        record["score"] = scores.get(key)
        harness = durations.get(key)
        record["elapsed_seconds"] = (
            round(harness, 3) if harness is not None else record["episode_seconds"]
        )
        record["harness_overhead_seconds"] = round(
            max(record["elapsed_seconds"] - record["episode_seconds"], 0.0), 3
        )

    summary = _summarise(
        rollouts, status, kind="decomposer", population=len(population)
    )
    summary["clamped_rollout_durations"] = clamped
    summary["manager_backend"] = manager_backend
    # A remote manager burns no local GPU, so this run's GPU-hours cover the
    # worker alone and understate the system against locally served rows.
    summary["gpu_hours_cover_manager"] = manager_backend == "local_vllm"
    manager_input_total = sum(r["manager_input_tokens"] for r in rollouts)
    floor_each = _mean([r["subagent_input_floor_tokens"] for r in rollouts])
    summary["tokens"] = {
        "method": "subagent totals measured from the worker vLLM log, calibrated "
        "on the manager's exact usage; the reconstruction is a floor",
        "sampled_fraction": round(sampled, 4),
        "integral_calibration": round(calibration, 4),
        # "none" means the manager ran on a remote API: there is no local log to
        # measure the counter's bias against, so the worker integral is raw.
        "calibration_source": calibration_source,
        "mean_assistant_tokens_per_subagent_call": round(mean_assistant, 1),
        "worker_vllm_integral": worker.as_dict(),
        "manager_vllm_integral": manager_measured.as_dict() if manager_measured else None,
        "manager_output_exact_per_rollout": round(manager_output_total / len(rollouts)),
        "manager_input_exact_per_rollout": round(manager_input_total / len(rollouts)),
        "subagent_input_source": input_source,
        "subagent_input_floor_per_rollout": round(floor_each),
        "subagent_input_floor_vs_measured": (
            round(floor_each / subagent_input_each, 4) if subagent_input_each else None
        ),
        "subagent_calls_counted_per_rollout": round(subagent_calls / len(rollouts), 2),
        "subagent_calls_implied_per_rollout": (
            round(subagent_input_each / rollouts[0]["schema_tokens"], 1)
            if rollouts and rollouts[0].get("schema_tokens")
            else None
        ),
    }
    return {"summary": summary, "rollouts": rollouts}


def _subagent_system_prompt() -> str:
    from decomposer.prompts import SUBAGENT_SYSTEM_PROMPT

    return SUBAGENT_SYSTEM_PROMPT


def analyse_simple_run(run_dir: Path, limit: int | None = None) -> dict[str, Any]:
    """Aggregate one direct-ReAct baseline run from its lite traces."""
    status = json.loads((run_dir / "run_status.json").read_text())
    year = datetime.fromisoformat(status["finished_at"]).year
    model_path = _simple_model_path(status)
    tokens = TokenCounter(model_path)

    population = sorted((run_dir / "lite").glob("*.json"))
    traces = population[:limit] if limit else population
    ceiling = _wall_seconds(status)
    clamped = 0
    schema_tokens = _domain_schema_tokens(
        status.get("domain") or "", tokens, run_dir.parent.parent
    )

    rollouts: list[dict[str, Any]] = []
    for path in traces:
        data = json.loads(path.read_text())
        duration = float(data.get("run_duration") or 0.0)
        if ceiling and duration > ceiling:
            duration = ceiling
            clamped += 1
        histories = data.get("per_agent_interaction_histories") or {}
        usage = data.get("per_agent_llm_usage_stats") or {}
        calls = sum(int((v or {}).get("total_llm_calls") or 0) for v in usage.values())
        input_tokens = output_tokens = 0
        for messages in histories.values():
            counts = [
                tokens.count(m.get("content")) + tokens.count(m.get("reasoning_content"))
                for m in messages
            ]
            running = schema_tokens
            for index, message in enumerate(messages):
                if message.get("role") == "assistant":
                    input_tokens += running
                    output_tokens += counts[index]
                running += counts[index]
        rollouts.append(
            {
                "scenario_id": data.get("scenario_id"),
                "run_number": _run_number(path.name),
                "turns": 1,
                "manager_failures": [],
                "elapsed_seconds": round(duration, 3),
                "episode_seconds": round(duration, 3),
                "harness_overhead_seconds": 0.0,
                "manager_only_seconds": round(duration, 3),
                "subagent_queue_seconds": 0.0,
                "env_tool_seconds": 0.0,
                "env_tool_calls": sum(
                    1 for ms in histories.values() for m in ms if m.get("role") == "tool-response"
                ),
                "spawns": 0,
                "subagent_statuses": {},
                "manager_input_floor_tokens": input_tokens,
                "manager_input_tokens": input_tokens,
                "manager_output_tokens": output_tokens,
                "manager_reasoning_tokens": 0,
                "manager_cached_input_tokens": 0,
                "manager_calls": calls,
                "subagent_calls": 0,
                "subagent_input_tokens": 0,
                "subagent_output_tokens": 0,
                "total_tokens": input_tokens + output_tokens,
                "structural": {},
                "temporal": {},
                "tool_call_join": {},
            }
        )

    scores = _outcomes(run_dir)
    for record in rollouts:
        record["score"] = scores.get((record["scenario_id"], record["run_number"]))

    policy = parse_vllm_throughput(run_dir / "logs" / "policy_vllm.log", year)
    # The message history holds every assistant token, so output is exact and
    # calibrates the throughput counter for this very server. Input is not
    # exact: the tool schemas travel in the request's `tools` field, never in a
    # message, so the reconstruction is a floor and the calibrated integral is
    # the figure to compare against the decomposer.
    output_exact = sum(r["manager_output_tokens"] for r in rollouts)
    calibration = 1.0
    if policy.generation_tokens and rollouts:
        predicted = output_exact / len(rollouts) * len(population)
        calibration = predicted / policy.generation_tokens
    logical = policy.logical_prompt_tokens
    measured_each = (
        logical * calibration / len(population) if logical and population else None
    )
    input_each = _mean([r["manager_input_floor_tokens"] for r in rollouts])
    for record in rollouts:
        record["total_tokens"] = (
            record["manager_input_tokens"] + record["manager_output_tokens"]
        )
    summary = _summarise(rollouts, status, kind="simple", population=len(population))
    summary["clamped_rollout_durations"] = clamped
    summary["manager_backend"] = "local_vllm"
    summary["gpu_hours_cover_manager"] = True
    summary["manager_backend"] = manager_backend
    # A remote manager burns no local GPU, so this run's GPU-hours cover the
    # worker alone and understate the system against locally served rows.
    summary["gpu_hours_cover_manager"] = manager_backend == "local_vllm"
    summary["tokens"] = {
        "method": "input reconstructed from the message history plus the tool "
        "schemas the benchmark binds per call; the calibrated throughput "
        "measurement corroborates it where the prefix-cache hit rate allows",
        "integral_calibration": round(calibration, 4),
        # "none" means the manager ran on a remote API: there is no local log to
        # measure the counter's bias against, so the worker integral is raw.
        "calibration_source": calibration_source,
        "policy_vllm_integral": policy.as_dict(),
        "output_exact_per_rollout": round(output_exact / max(len(rollouts), 1)),
        "schema_tokens_per_call": schema_tokens,
        "input_source": "reconstruction",
        "input_measured_per_rollout": round(measured_each) if measured_each else None,
        "input_floor_vs_measured": (
            round(input_each / measured_each, 4) if measured_each else None
        ),
    }
    return {"summary": summary, "rollouts": rollouts}


def _domain_schema_tokens(domain: str, tokens: TokenCounter, root: Path) -> int:
    """Tokens for the scenario tool schemas the benchmark binds to an agent.

    The simple agent's tools travel in the request's ``tools`` field, so they
    never appear in the message history the lite trace records: reconstructing
    from messages alone omits them entirely and undercounts the direct baseline
    by roughly half. The schemas are recoverable from any decomposer sidecar in
    the same domain, which stores them per episode. That set excludes the few
    AgentUserInterface tools the decomposer's broker hides from its workers, so
    the figure is marginally low.
    """
    for run_dir in sorted((root / domain).iterdir()):
        sidecars = sorted((run_dir / "decomposer_sidecars").glob("*.json"))[:1]
        for path in sidecars:
            data = json.loads(path.read_text())
            for turn in data.get("turns") or []:
                schemas = (
                    (turn["manager"].get("trace") or {}).get("runtime_context") or {}
                ).get("tool_schemas")
                if schemas:
                    return tokens.count(
                        json.dumps(schemas, ensure_ascii=False, default=str)
                    )
    return 0


def _simple_model_path(status: dict[str, Any]) -> str | None:
    """The policy checkpoint, from the preparation manifest the run pinned.

    ``run_status.json`` records sampling and budgets but not the checkpoint; the
    manifest it points at hashes every model file and carries the snapshot path.
    """
    manifest = status.get("preparation_manifest")
    if not manifest:
        return None
    models = (json.loads(Path(manifest).read_text()).get("models") or {})
    for entry in models.values():
        if isinstance(entry, dict) and entry.get("path"):
            return entry["path"]
    return None


def _run_number(name: str) -> int | None:
    match = re.search(r"_run_(\d+)_", name)
    return int(match.group(1)) if match else None


def _summarise(
    rollouts: Sequence[dict[str, Any]],
    status: dict[str, Any],
    *,
    kind: str,
    population: int,
) -> dict[str, Any]:
    """Collapse per-rollout records into the numbers a cost table reports.

    ``population`` is every rollout the run produced; ``rollouts`` may be a
    sample. Per-rollout means come from the sample, wall-clock rates from the
    population, because the elapsed time covers all of it either way.
    """
    count = len(rollouts) or 1
    population = population or count
    gpus = len(status.get("cuda_visible_devices") or []) or 1
    concurrency = int(status.get("concurrency") or 1)
    wall = _wall_seconds(status)

    solved = [r for r in rollouts if (r.get("score") or 0.0) > 0]
    tokens_total = sum(r["total_tokens"] for r in rollouts)

    summary: dict[str, Any] = {
        "kind": kind,
        "experiment": status.get("experiment"),
        "domain": status.get("domain"),
        "prompt_profile": status.get("decomposer_system_prompt_profile"),
        "rollouts": len(rollouts),
        "population_rollouts": population,
        "empty_rollouts": sum(1 for r in rollouts if not r.get("turns", 1)),
        "manager_failures": dict(
            sorted(Counter(k for r in rollouts for k in r.get("manager_failures", [])).items())
        ),
        "solved_rollouts": len(solved),
        "concurrency": concurrency,
        "gpus": gpus,
        "wall_seconds": round(wall, 1),
        "wall_seconds_per_rollout": round(wall / population, 2),
        "gpu_hours_per_rollout": round(wall * gpus / population / 3600.0, 5),
        "rollouts_per_hour": round(population / (wall / 3600.0), 2) if wall else None,
        # Wall time spread over the slots the run actually had. Unlike the
        # per-rollout clock this cannot be wrong about what it measures, so it
        # is the figure to compare across systems.
        "slot_seconds_per_rollout": round(wall * concurrency / population, 2),
        # Per-rollout times summed and spread over the run's own concurrency
        # should reproduce its wall clock; a ratio far from 1.0 means the
        # per-rollout clock is measuring something narrower than the slot.
        "clock_reconciliation": (
            round(
                _mean([r["elapsed_seconds"] for r in rollouts])
                * population
                / concurrency
                / wall,
                4,
            )
            if wall and rollouts
            else None
        ),
        "latency": {
            "rollout_seconds": _percentiles([r["elapsed_seconds"] for r in rollouts]),
            "episode_seconds": _percentiles([r["episode_seconds"] for r in rollouts]),
            "harness_overhead_seconds": _percentiles(
                [r.get("harness_overhead_seconds", 0.0) for r in rollouts]
            ),
            "manager_only_seconds": _percentiles(
                [r["manager_only_seconds"] for r in rollouts]
            ),
            "subagent_queue_seconds": _percentiles(
                [r["subagent_queue_seconds"] for r in rollouts]
            ),
            "env_tool_seconds": _percentiles([r["env_tool_seconds"] for r in rollouts]),
        },
        "spawns": _percentiles([r["spawns"] for r in rollouts]),
        "env_tool_calls": _percentiles([r["env_tool_calls"] for r in rollouts]),
        "tokens_per_rollout": {
            "manager_input": round(sum(r["manager_input_tokens"] for r in rollouts) / count),
            "manager_output": round(sum(r["manager_output_tokens"] for r in rollouts) / count),
            "manager_reasoning": round(
                sum(r.get("manager_reasoning_tokens", 0) for r in rollouts) / count
            ),
            "manager_cached_input": round(
                sum(r.get("manager_cached_input_tokens", 0) for r in rollouts) / count
            ),
            "subagent_input": round(sum(r["subagent_input_tokens"] for r in rollouts) / count),
            "subagent_output": round(sum(r["subagent_output_tokens"] for r in rollouts) / count),
            "total": round(tokens_total / count),
        },
        "tokens_per_solved_rollout": (
            round(tokens_total / len(solved)) if solved else None
        ),
        "calls_per_rollout": {
            "manager": round(sum(r["manager_calls"] for r in rollouts) / count, 2),
            "subagent": round(sum(r["subagent_calls"] for r in rollouts) / count, 2),
        },
    }

    if kind == "decomposer":
        statuses: Counter[str] = Counter()
        for record in rollouts:
            statuses.update(record["subagent_statuses"])
        bursts = [b for r in rollouts for b in r["structural"].get("spawn_bursts", [])]
        summary["parallelism"] = {
            "structural": {
                "max_outstanding": _percentiles(
                    [r["structural"].get("max_outstanding", 0) for r in rollouts]
                ),
                "mean_outstanding_at_spawn": round(
                    _mean([r["structural"].get("mean_outstanding_at_spawn", 0.0) for r in rollouts]),
                    3,
                ),
                "reports_per_wait": round(
                    _mean([r["structural"].get("mean_reports_per_wait", 0.0) for r in rollouts]), 3
                ),
                "spawn_burst_histogram": dict(sorted(Counter(bursts).items())),
                "rollouts_with_any_batching": sum(
                    1 for r in rollouts if r["structural"].get("max_outstanding", 0) > 1
                ),
                "waits_per_rollout": round(
                    _mean([r["structural"].get("waits", 0) for r in rollouts]), 3
                ),
                # `wait` with nothing in flight: the manager blocking on work it
                # never started. Pure waste, so it is counted on its own.
                "empty_waits_per_rollout": round(
                    _mean([r["structural"].get("empty_waits", 0) for r in rollouts]), 3
                ),
                "spawn_errors_per_rollout": round(
                    _mean([r["structural"].get("spawn_errors", 0) for r in rollouts]), 3
                ),
            },
            "temporal": {
                "max_concurrent": _percentiles(
                    [r["temporal"].get("max_concurrent", 0) for r in rollouts]
                ),
                "mean_concurrent": round(
                    _mean([r["temporal"].get("mean_concurrent", 0.0) for r in rollouts]), 3
                ),
                "overlap_factor": round(
                    _mean([r["temporal"].get("overlap_factor", 0.0) for r in rollouts]), 3
                ),
                "busy_union_seconds": _percentiles(
                    [r["temporal"].get("busy_union_seconds", 0.0) for r in rollouts]
                ),
            },
            "subagent_statuses": dict(statuses),
        }
        join: Counter[str] = Counter()
        for record in rollouts:
            join.update(record["tool_call_join"])
        total_join = sum(join.values()) or 1
        summary["tool_call_join"] = {
            **dict(join),
            "unique_fraction": round(join.get("unique", 0) / total_join, 4),
        }
    return summary


# --------------------------------------------------------------------------
# run discovery and CLI
# --------------------------------------------------------------------------

WORKERS = ["e2b", "e4b", "26b-a4b"]
DOMAINS = ["search", "execution"]
SYSTEMS: dict[str, tuple[str, str]] = {
    # label -> (glob stem, suffix)
    "Direct ReAct": ("gemma4-{worker}-it-non-thinking-n3", ""),
    "Mos'ka SFT / teacher": ("qwen35-4b-sft-mixed-v3-non-thinking-gemma4-{worker}-non-thinking-n3", ""),
    "Mos'ka SFT / student": ("qwen35-4b-sft-student-non-thinking-gemma4-{worker}-non-thinking-n3", ""),
    "Untuned 4B / student": (
        "qwen35-4b-base-non-thinking-gemma4-{worker}-non-thinking-n3",
        "-prompt-student",
    ),
    "Untuned 4B / teacher": (
        "qwen35-4b-base-non-thinking-teacher-gemma4-{worker}-non-thinking-n3",
        "",
    ),
    # occ-cds only: the manager is a remote reasoning API, so these runs carry no
    # manager vLLM log and only one local GPU (the worker).
    "DeepSeek teacher": (
        "deepseek-v4-flash-0731-reasoning-max-teacher-gemma4-{worker}-non-thinking-n3",
        "-prompt-teacher",
    ),
}


def resolve_run(
    system: str, worker: str, domain: str, root: Path = RESULTS_ROOT
) -> Path | None:
    """Locate a run directory the way the accuracy table does.

    Re-runs land in ``-port-offset-N`` directories and superseded runs are
    renamed ``.INVALID-*``; a cell with more than one clean candidate is refused
    rather than guessed at.
    """
    stem, suffix = SYSTEMS[system]
    pattern = f"{stem.format(worker=worker)}{suffix}*"
    candidates = [
        c
        for c in sorted((root / domain).glob(pattern))
        if "INVALID" not in c.name and c.is_dir()
    ]
    if system == "Untuned 4B / teacher":
        candidates = [c for c in candidates if "prompt-student" not in c.name]
    if system == "Untuned 4B / student":
        candidates = [c for c in candidates if "teacher" not in c.name]
    return candidates[0] if len(candidates) == 1 else None


def analyse(run_dir: Path, limit: int | None = None) -> dict[str, Any]:
    if (run_dir / "decomposer_sidecars").is_dir():
        return analyse_decomposer_run(run_dir, limit=limit)
    return analyse_simple_run(run_dir, limit=limit)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAINS, action="append")
    parser.add_argument("--system", choices=sorted(SYSTEMS), action="append")
    parser.add_argument("--worker", choices=WORKERS, action="append")
    parser.add_argument("--run-dir", type=Path, help="analyse one directory directly")
    parser.add_argument("--limit", type=int, help="only the first N rollouts (smoke)")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="benchmark results tree (occ-cds holds the DeepSeek runs elsewhere)",
    )
    parser.add_argument("--out", type=Path, default=ANALYSIS_ROOT)
    args = parser.parse_args()

    jobs: list[tuple[str, str, str, Path]] = []
    if args.run_dir:
        jobs.append(("direct", "-", args.run_dir.parent.name, args.run_dir))
    else:
        for domain in args.domain or DOMAINS:
            for system in args.system or sorted(SYSTEMS):
                for worker in args.worker or WORKERS:
                    found = resolve_run(system, worker, domain, args.results_root)
                    if found is None:
                        print(f"  [skip] {domain:<9} {system:<22} {worker:<8} unresolved")
                        continue
                    jobs.append((system, worker, domain, found))

    for system, worker, domain, run_dir in jobs:
        print(f"  [run ] {domain:<9} {system:<22} {worker:<8} {run_dir.name}")
        result = analyse(run_dir, limit=args.limit)
        target = args.out / domain
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{run_dir.name}.json").write_text(
            json.dumps({"system": system, "worker": worker, **result["summary"]}, indent=2)
        )
        with (target / f"{run_dir.name}.jsonl").open("w") as handle:
            for record in result["rollouts"]:
                handle.write(json.dumps(record) + "\n")
        summary = result["summary"]
        print(
            f"         rollouts={summary['rollouts']} "
            f"spawns={summary['spawns']['mean']} "
            f"tok/rollout={summary['tokens_per_rollout']['total']} "
            f"s/rollout={summary['latency']['rollout_seconds']['mean']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
