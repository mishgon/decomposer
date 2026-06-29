"""CLI commands for hierarchical decomposer runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .cli import (
    _append_grading_to_trace,
    _collect_env_snapshot,
    _grade_with_optional_params,
    _make_judge,
    _make_trace_dir,
    _resolve_task_yaml,
    _resolve_tasks_dir,
    _resolve_batch_task_dirs,
    _save_env_snapshot,
    _scan_completed_trials,
    _trace_totals,
    _apply_proxy,
    _split_vllm_extra_args,
)
from .config import Config, DecomposerRunConfig, ModelConfig, load_config
from .graders.registry import get_grader
from .models.scoring import compute_pass_at_k, compute_pass_hat_k, compute_task_score, is_pass
from .models.task import TaskDefinition
from .runner.decomposer import run_decomposer_task
from .runner.providers.openai_compat import OpenAICompatProvider
from .runner.services import ServiceManager
from .trace.reader import load_trace, load_trace_for_grading


def _make_decomposer_trace_dir(
    base_dir: str | Path,
    decomposer_model: str,
    executor_model: str,
) -> Path:
    from datetime import datetime

    date_str = datetime.now().strftime("%y-%m-%d-%H-%M")
    safe_decomp = decomposer_model.replace("/", "_")
    safe_exec = executor_model.replace("/", "_")
    trace_dir = Path(base_dir) / f"{safe_decomp}__{safe_exec}_{date_str}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def _resolve_executor_model_cfg(cfg: Config, args: argparse.Namespace) -> ModelConfig:
    base = cfg.executor_model.model_copy() if cfg.executor_model else cfg.model.model_copy()
    if getattr(args, "executor_model", None):
        base.model_id = args.executor_model
    if getattr(args, "executor_api_key", None):
        base.api_key = args.executor_api_key
    elif getattr(args, "api_key", None):
        base.api_key = args.api_key
    if getattr(args, "executor_base_url", None):
        base.base_url = args.executor_base_url
    elif getattr(args, "base_url", None):
        base.base_url = args.base_url
    return base


def _resolve_decomposer_model_cfg(cfg: Config, args: argparse.Namespace) -> ModelConfig:
    base = cfg.model.model_copy()
    if getattr(args, "decomposer_model", None):
        base.model_id = args.decomposer_model
    elif getattr(args, "model", None):
        base.model_id = args.model
    if getattr(args, "decomposer_api_key", None):
        base.api_key = args.decomposer_api_key
    elif getattr(args, "api_key", None):
        base.api_key = args.api_key
    if getattr(args, "decomposer_base_url", None):
        base.base_url = args.decomposer_base_url
    elif getattr(args, "base_url", None):
        base.base_url = args.base_url
    return base


def _make_providers(cfg: Config, args: argparse.Namespace) -> tuple[OpenAICompatProvider, OpenAICompatProvider, ModelConfig, ModelConfig]:
    decomposer_cfg = _resolve_decomposer_model_cfg(cfg, args)
    executor_cfg = _resolve_executor_model_cfg(cfg, args)
    decomposer_provider = OpenAICompatProvider(
        model_id=decomposer_cfg.model_id,
        api_key=decomposer_cfg.api_key,
        base_url=decomposer_cfg.base_url,
        extra_body=decomposer_cfg.extra_body,
        temperature=decomposer_cfg.temperature,
        reasoning_effort=decomposer_cfg.reasoning_effort,
    )
    executor_provider = OpenAICompatProvider(
        model_id=executor_cfg.model_id,
        api_key=executor_cfg.api_key,
        base_url=executor_cfg.base_url,
        extra_body=executor_cfg.extra_body,
        temperature=executor_cfg.temperature,
        reasoning_effort=executor_cfg.reasoning_effort,
    )
    return decomposer_provider, executor_provider, decomposer_cfg, executor_cfg


def _maybe_launch_decomposer_vllm(
    args: argparse.Namespace,
    cfg: Config,
    *,
    decomposer_model: str,
    executor_model: str,
):
    """Start/reuse decomposer and executor vLLM servers when requested."""

    if not getattr(args, "launch_vllm", False):
        return None
    from .runner.vllm_process import ensure_vllm_servers, make_vllm_server_spec

    executor_cfg = cfg.executor_model or cfg.model
    common = {
        "host": getattr(args, "vllm_host", "127.0.0.1"),
        "max_model_len": getattr(args, "vllm_max_model_len", None),
        "gpu_memory_utilization": getattr(args, "vllm_gpu_memory_utilization", None),
        "extra_args": _split_vllm_extra_args(getattr(args, "vllm_extra_arg", None)),
    }
    specs = [
        make_vllm_server_spec(
            role="decomposer",
            model_id=decomposer_model,
            port=getattr(args, "decomposer_port", 8000),
            gpu=getattr(args, "decomposer_gpu", "0"),
            api_key=getattr(args, "api_key", None) or cfg.model.api_key or "unused",
            **common,
        ),
        make_vllm_server_spec(
            role="executor",
            model_id=executor_model,
            port=getattr(args, "executor_port", 8001),
            gpu=getattr(args, "executor_gpu", "1"),
            api_key=getattr(args, "api_key", None) or executor_cfg.api_key or "unused",
            **common,
        ),
    ]
    return ensure_vllm_servers(
        specs,
        log_dir=getattr(args, "vllm_log_dir", "logs/vllm"),
        timeout_s=cfg.vllm.ready_timeout_s,
        poll_s=cfg.vllm.ready_poll_s,
        check_health=cfg.vllm.check_health,
        stop_on_exit=getattr(args, "stop_vllm_on_exit", False),
    )


def _warn_sandbox_tools(task: TaskDefinition, sandbox_tools: bool) -> None:
    if not task.tools and not sandbox_tools:
        print(
            f"[WARNING] Task {task.task_id} has no task tools; "
            "use --sandbox-tools or --sandbox for terminal/sandbox tasks."
        )


def _grade_trace(
    trace_path: Path,
    task: TaskDefinition,
    task_yaml: Path,
    tasks_dir: Path,
    cfg: Config,
    args: argparse.Namespace,
    env_snapshot: dict | None,
) -> tuple[float, bool, object]:
    judge = _make_judge(cfg, args)
    start, messages, dispatches, media_events, end, audit_data = load_trace_for_grading(trace_path)
    grader = get_grader(task.task_id, tasks_dir=tasks_dir, task_dir=task_yaml.parent)
    scores, judge_calls = _grade_with_optional_params(
        grader, messages, dispatches, task,
        audit_data=audit_data, judge=judge, media_events=media_events,
        env_snapshot=env_snapshot,
    )
    task_score = compute_task_score(scores)
    passed = is_pass(task_score)
    _append_grading_to_trace(
        trace_path,
        trace_id=start.trace_id,
        task_id=task.task_id,
        scores=scores,
        task_score=task_score,
        passed=passed,
        judge_calls=judge_calls,
    )
    return task_score, passed, scores


def _print_grade_scores(scores, task_score: float, passed: bool, end) -> None:
    totals = _trace_totals(end)
    print(f"  completion:     {scores.completion:.2f}")
    print(f"  robustness:     {scores.robustness:.2f}")
    print(f"  communication:  {scores.communication:.2f}")
    print(f"  safety:         {scores.safety:.1f}")
    print(f"  task_score:     {task_score:.2f}")
    print(f"  passed:         {passed}")
    print(
        f"  model_tokens:   {totals['total_tokens']} "
        f"({totals['model_input_tokens']} in / {totals['model_output_tokens']} out)"
    )
    print(
        f"  time_s:         wall={totals['wall_time_s']:.2f} "
        f"model={totals['model_time_s']:.2f} tool={totals['tool_time_s']:.2f} "
        f"other={totals['other_time_s']:.2f}"
    )


def _load_local_grader_files(task: TaskDefinition, task_yaml: Path, env_snapshot: dict | None) -> dict | None:
    if not task.local_grader_files:
        return env_snapshot
    import base64 as _b64

    if env_snapshot is None:
        env_snapshot = {}
    task_root = Path(str(task_yaml.parent))
    for rel_path in task.local_grader_files:
        local_path = task_root / rel_path
        if local_path.exists():
            content = _b64.b64encode(local_path.read_bytes()).decode()
            env_snapshot[f"local_file:{rel_path}"] = {
                "encoding": "base64",
                "content": content,
            }
        else:
            env_snapshot[f"local_file:{rel_path}"] = {
                "error": f"not found: {local_path}",
            }
    return env_snapshot


def cmd_run_decomposer(args: argparse.Namespace) -> None:
    """Run hierarchical decomposer+executor on a task."""
    _apply_proxy(getattr(args, "proxy", None))
    cfg = load_config(args.config)
    decomposer_model = args.decomposer_model or args.model or cfg.model.model_id
    executor_model = args.executor_model or (cfg.executor_model.model_id if cfg.executor_model else cfg.model.model_id)
    vllm_group = _maybe_launch_decomposer_vllm(
        args,
        cfg,
        decomposer_model=decomposer_model,
        executor_model=executor_model,
    )
    if vllm_group is not None:
        import atexit

        atexit.register(vllm_group.close)
        args.no_vllm_wait = True
        cfg = load_config(args.config)

    from .runner.vllm_cli import maybe_wait_for_vllm

    maybe_wait_for_vllm(
        cfg,
        args,
        roles=["decomposer", "executor"],
        cli_overrides={
            "decomposer": getattr(args, "decomposer_model", None) or getattr(args, "model", None),
            "executor": getattr(args, "executor_model", None),
        },
        include_judge=not getattr(args, "skip_grade", False),
    )

    task_yaml = _resolve_task_yaml(args.task)
    task = TaskDefinition.from_yaml(task_yaml)
    tasks_dir = _resolve_tasks_dir(task_yaml)

    port_offset = getattr(args, "port_offset", 0) or 0
    if port_offset:
        task.apply_port_offset(port_offset)

    decomposer_provider, executor_provider, decomposer_model_cfg, executor_model_cfg = _make_providers(cfg, args)
    base_trace_dir = args.trace_dir or cfg.defaults.trace_dir
    if args.trace_dir:
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
    else:
        trace_dir = _make_decomposer_trace_dir(
            base_trace_dir,
            decomposer_provider.model_id,
            executor_provider.model_id,
        )

    sandbox_tools = getattr(args, "sandbox_tools", False)
    sandbox_mode = (getattr(args, "sandbox", False) or cfg.sandbox.enabled) and not getattr(args, "no_sandbox", False)
    _warn_sandbox_tools(task, sandbox_tools or sandbox_mode)
    trials = args.trials or 1
    trial_scores: list[float] = []
    trace_paths: list[Path] = []

    decomposer_run_cfg = cfg.decomposer

    if sandbox_mode:
        from .runner.sandbox_runner import SandboxRunner

        sandbox_image = getattr(args, "sandbox_image", None) or cfg.sandbox.image
        runner = SandboxRunner(cfg.sandbox, image=sandbox_image)

        with ServiceManager(task.services, mock_today=task.environment.mock_today) as svc:
            for i in range(trials):
                if trials > 1:
                    print(f"\n--- Trial {i + 1}/{trials} ---")
                if i > 0:
                    svc.reset_all()

                run_id = f"{task.task_id}-decomp-trial{i}"
                handle = runner.start_container(run_id=run_id)
                env_snapshot = None
                try:
                    runner.inject_files(handle, task, task_dir=str(task_yaml.parent))
                    trace_path = run_decomposer_task(
                        task,
                        decomposer_provider,
                        executor_provider,
                        trace_dir=trace_dir,
                        decomposer_cfg=decomposer_run_cfg,
                        sandbox_tools=True,
                        sandbox_url=handle.sandbox_url,
                        prompt_cfg=cfg.prompt,
                        decomposer_model_cfg=decomposer_model_cfg,
                        executor_model_cfg=executor_model_cfg,
                        media_cfg=cfg.media,
                        react_cfg=cfg.react,
                    )
                    runner.inject_grader_files(handle, task, task_dir=str(task_yaml.parent))
                    env_snapshot = _collect_env_snapshot(handle.sandbox_url, task)
                    _save_env_snapshot(env_snapshot, trace_path, task.task_id)
                finally:
                    runner.stop_container(handle)

                env_snapshot = _load_local_grader_files(task, task_yaml, env_snapshot)
                trace_paths.append(trace_path)
                print(f"Trace: {trace_path}")
                if getattr(args, "skip_grade", False):
                    print("  [skip-grade] Trace written; skipping grading for later.")
                    continue
                task_score, passed, scores = _grade_trace(
                    trace_path, task, task_yaml, tasks_dir, cfg, args, env_snapshot,
                )
                trial_scores.append(task_score)
                _, _, _, _, end, _ = load_trace(trace_path)
                _print_grade_scores(scores, task_score, passed, end)
    else:
        with ServiceManager(task.services, mock_today=task.environment.mock_today) as svc:
            for i in range(trials):
                if trials > 1:
                    print(f"\n--- Trial {i + 1}/{trials} ---")
                if i > 0:
                    svc.reset_all()

                trace_path = run_decomposer_task(
                    task,
                    decomposer_provider,
                    executor_provider,
                    trace_dir=trace_dir,
                    decomposer_cfg=decomposer_run_cfg,
                    sandbox_tools=sandbox_tools,
                    prompt_cfg=cfg.prompt,
                    decomposer_model_cfg=decomposer_model_cfg,
                    executor_model_cfg=executor_model_cfg,
                    media_cfg=cfg.media,
                    react_cfg=cfg.react,
                )
                trace_paths.append(trace_path)
                print(f"Trace: {trace_path}")
                if getattr(args, "skip_grade", False):
                    print("  [skip-grade] Trace written; skipping grading for later.")
                    continue
                env_snapshot = _load_local_grader_files(task, task_yaml, None)
                task_score, passed, scores = _grade_trace(
                    trace_path, task, task_yaml, tasks_dir, cfg, args, env_snapshot,
                )
                trial_scores.append(task_score)
                _, _, _, _, end, _ = load_trace(trace_path)
                _print_grade_scores(scores, task_score, passed, end)

    if trials > 1 and trial_scores:
        print(f"\n--- Multi-trial summary ({trials} trials) ---")
        for i, (score, path) in enumerate(zip(trial_scores, trace_paths)):
            print(f"  Trial {i+1}: score={score:.2f} pass={is_pass(score)} trace={path}")
        print(f"  pass@1:  {compute_pass_at_k(trial_scores, k=1):.3f}")
        print(f"  pass^{trials}:  {compute_pass_hat_k(trial_scores, k=trials):.3f}")


def _run_single_decomposer_task(
    task_dir: str,
    config_path: str | None,
    decomposer_model: str | None,
    executor_model: str | None,
    api_key: str | None,
    base_url: str | None,
    trace_dir: str | None,
    port_offset: int,
    no_judge: bool,
    judge_model: str | None,
    trials: int,
    proxy: str | None = None,
    sandbox: bool = False,
    sandbox_image: str | None = None,
    sandbox_tools: bool = False,
    no_vllm_wait: bool = False,
    skip_grade: bool = False,
    no_sandbox: bool = False,
) -> dict:
    """Run a single decomposer task in a worker process."""
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    _apply_proxy(proxy)

    task_yaml = Path(task_dir) / "task.yaml"
    if not task_yaml.exists():
        task_yaml = Path(task_dir)
    task = TaskDefinition.from_yaml(task_yaml)
    tasks_dir = task_yaml.parent.parent
    task_dir_path = str(task_yaml.parent)

    if port_offset:
        task.apply_port_offset(port_offset)

    cfg = load_config(config_path)
    args_ns = argparse.Namespace(
        no_judge=no_judge or skip_grade,
        judge_model=judge_model,
        decomposer_model=decomposer_model,
        executor_model=executor_model,
        api_key=api_key,
        base_url=base_url,
        no_vllm_wait=no_vllm_wait,
    )

    from .runner.vllm_cli import maybe_wait_for_vllm

    maybe_wait_for_vllm(
        cfg,
        args_ns,
        roles=["decomposer", "executor"],
        cli_overrides={
            "decomposer": decomposer_model,
            "executor": executor_model,
        },
        include_judge=True,
    )

    decomposer_provider, executor_provider, decomposer_model_cfg, executor_model_cfg = _make_providers(cfg, args_ns)
    judge = None if skip_grade else _make_judge(cfg, args_ns)

    sandbox_mode = (sandbox or cfg.sandbox.enabled) and not no_sandbox
    sandbox_runner = None
    if sandbox_mode:
        from .runner.sandbox_runner import SandboxRunner
        sandbox_runner = SandboxRunner(cfg.sandbox, image=sandbox_image or cfg.sandbox.image)

    result = {
        "task_id": task.task_id,
        "task_name": task.task_name,
        "difficulty": task.difficulty,
        "run_mode": "decomposer",
        "decomposer_model": decomposer_provider.model_id,
        "executor_model": executor_provider.model_id,
        "trials": [],
        "error": None,
    }

    from openai import APIConnectionError, APITimeoutError, InternalServerError

    max_retries = 3
    for attempt in range(max_retries):
        result["trials"] = []
        try:
            with ServiceManager(task.services, cwd=tasks_dir.parent, mock_today=task.environment.mock_today) as svc:
                for i in range(trials):
                    if i > 0:
                        svc.reset_all()
                    try:
                        env_snapshot = None
                        if sandbox_runner:
                            run_id = f"{task.task_id}-decomp-t{i}-p{port_offset}"
                            handle = sandbox_runner.start_container(run_id=run_id)
                            try:
                                sandbox_runner.inject_files(handle, task, task_dir=task_dir_path)
                                trace_path = run_decomposer_task(
                                    task,
                                    decomposer_provider,
                                    executor_provider,
                                    trace_dir=trace_dir or cfg.defaults.trace_dir,
                                    decomposer_cfg=cfg.decomposer,
                                    sandbox_tools=True,
                                    sandbox_url=handle.sandbox_url,
                                    prompt_cfg=cfg.prompt,
                                    decomposer_model_cfg=decomposer_model_cfg,
                                    executor_model_cfg=executor_model_cfg,
                                    media_cfg=cfg.media,
                                    react_cfg=cfg.react,
                                )
                                sandbox_runner.inject_grader_files(handle, task, task_dir=task_dir_path)
                                env_snapshot = _collect_env_snapshot(handle.sandbox_url, task)
                                _save_env_snapshot(env_snapshot, trace_path, task.task_id)
                            finally:
                                sandbox_runner.stop_container(handle)
                        else:
                            trace_path = run_decomposer_task(
                                task,
                                decomposer_provider,
                                executor_provider,
                                trace_dir=trace_dir or cfg.defaults.trace_dir,
                                decomposer_cfg=cfg.decomposer,
                                sandbox_tools=sandbox_tools,
                                prompt_cfg=cfg.prompt,
                                decomposer_model_cfg=decomposer_model_cfg,
                                executor_model_cfg=executor_model_cfg,
                                media_cfg=cfg.media,
                                react_cfg=cfg.react,
                            )

                        if skip_grade:
                            _, _, _, _, end, _ = load_trace(trace_path)
                            totals = _trace_totals(end)
                            result["trials"].append({
                                "trace": str(trace_path),
                                "model_input_tokens": totals["model_input_tokens"],
                                "model_output_tokens": totals["model_output_tokens"],
                                "tokens": totals["total_tokens"],
                                "model_time_s": totals["model_time_s"],
                                "tool_time_s": totals["tool_time_s"],
                                "wall_time_s": totals["wall_time_s"],
                                "task_score": None,
                                "passed": None,
                                "skip_grade": True,
                            })
                            continue

                        env_snapshot = _load_local_grader_files(task, task_yaml, env_snapshot)
                        start, messages, dispatches, media_events, end, audit_data = load_trace_for_grading(trace_path)
                        grader = get_grader(task.task_id, tasks_dir=tasks_dir, task_dir=task_yaml.parent)
                        scores, judge_calls = _grade_with_optional_params(
                            grader, messages, dispatches, task,
                            audit_data=audit_data, judge=judge, media_events=media_events,
                            env_snapshot=env_snapshot,
                        )
                        task_score = compute_task_score(scores)
                        _append_grading_to_trace(
                            trace_path,
                            trace_id=start.trace_id,
                            task_id=task.task_id,
                            scores=scores,
                            task_score=task_score,
                            passed=is_pass(task_score),
                            judge_calls=judge_calls,
                        )
                        totals = _trace_totals(end)
                        result["trials"].append({
                            "trace": str(trace_path),
                            "model_input_tokens": totals["model_input_tokens"],
                            "model_output_tokens": totals["model_output_tokens"],
                            "tokens": totals["total_tokens"],
                            "model_time_s": totals["model_time_s"],
                            "tool_time_s": totals["tool_time_s"],
                            "wall_time_s": totals["wall_time_s"],
                            "completion": scores.completion,
                            "robustness": scores.robustness,
                            "communication": scores.communication,
                            "safety": scores.safety,
                            "task_score": task_score,
                            "passed": is_pass(task_score),
                        })
                    except Exception as trial_exc:
                        result["trials"].append({
                            "trial": i,
                            "error": str(trial_exc),
                            "task_score": 0.0,
                            "passed": False,
                        })
            break
        except (APIConnectionError, APITimeoutError, InternalServerError, ConnectionError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [{task.task_id}] retry {attempt + 1}/{max_retries} after {type(e).__name__}, waiting {wait}s")
                time.sleep(wait)
            else:
                result["error"] = str(e)
        except Exception as e:
            result["error"] = str(e)
            break

    valid_trials = [t for t in result["trials"] if not t.get("error") and not t.get("skip_grade")]
    trial_scores = [t["task_score"] for t in valid_trials]
    n_trials = len(trial_scores)
    if n_trials > 0:
        result["avg_score"] = sum(trial_scores) / n_trials
        result["pass_at_1"] = compute_pass_at_k(trial_scores, k=1)
        result["pass_hat_k"] = compute_pass_hat_k(trial_scores, k=n_trials)
        result["avg_passed"] = is_pass(result["avg_score"])
    else:
        if result["trials"] and all(t.get("skip_grade") for t in result["trials"]):
            result["grading_skipped"] = True
            result["avg_score"] = None
            result["pass_at_1"] = None
            result["pass_hat_k"] = None
            result["avg_passed"] = None
        else:
            result["avg_score"] = 0.0
            result["pass_at_1"] = 0.0
            result["pass_hat_k"] = 0.0
            result["avg_passed"] = False
        if result["trials"] and not result["error"] and not result.get("grading_skipped"):
            result["error"] = result["trials"][0].get("error", "all trials errored")

    return result


def cmd_batch_decomposer(args: argparse.Namespace) -> None:
    """Run decomposer mode on all (or filtered) tasks in parallel."""
    _apply_proxy(getattr(args, "proxy", None))

    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        print(f"Tasks directory not found: {tasks_dir}")
        sys.exit(1)

    try:
        task_dirs = _resolve_batch_task_dirs(args, tasks_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    if args.filter:
        filt = args.filter.lower()
        task_dirs = [d for d in task_dirs if filt in d.lower()]

    tag = getattr(args, "tag", None) or getattr(args, "split", None)
    if tag:
        filtered = []
        for d in task_dirs:
            td = TaskDefinition.from_yaml(Path(d) / "task.yaml")
            if tag in td.tags:
                filtered.append(d)
        task_dirs = filtered

    if getattr(args, "range", None):
        import re as _re
        _m = _re.match(r"(\d+)-(\d+)$", args.range)
        if not _m:
            print(f"[ERROR] Invalid --range format: {args.range}")
            sys.exit(1)
        lo, hi = int(_m.group(1)), int(_m.group(2))

        def _in_range(d: str) -> bool:
            name = Path(d).name
            m = _re.match(r"T(\d+)", name)
            return m is not None and lo <= int(m.group(1)) <= hi

        task_dirs = [d for d in task_dirs if _in_range(d)]

    if not task_dirs:
        print("No tasks matched.")
        return

    workers = args.parallel
    trials = args.trials or 1
    cfg = load_config(args.config)
    decomposer_model = args.decomposer_model or args.model or cfg.model.model_id
    executor_model = args.executor_model or (cfg.executor_model.model_id if cfg.executor_model else cfg.model.model_id)
    vllm_group = _maybe_launch_decomposer_vllm(
        args,
        cfg,
        decomposer_model=decomposer_model,
        executor_model=executor_model,
    )
    if vllm_group is not None:
        import atexit

        atexit.register(vllm_group.close)
        args.no_vllm_wait = True
        cfg = load_config(args.config)
        decomposer_model = args.decomposer_model or args.model or cfg.model.model_id
        executor_model = args.executor_model or (cfg.executor_model.model_id if cfg.executor_model else cfg.model.model_id)
    base_trace_dir = args.trace_dir or cfg.defaults.trace_dir
    batch_trace_dir = str(_make_decomposer_trace_dir(base_trace_dir, decomposer_model, executor_model))

    print(f"Running {len(task_dirs)} tasks (decomposer mode) with {workers} workers, {trials} trial(s) each")
    print(f"Decomposer: {decomposer_model}")
    print(f"Executor:   {executor_model}")
    print(f"Traces → {batch_trace_dir}\n")

    results: list[dict] = []
    start_time = time.monotonic()
    port_base_offset = getattr(args, "port_base_offset", 0)
    _STRIDE = 50

    with ProcessPoolExecutor(max_workers=workers) as pool:
        available_slots = list(range(workers))
        pending: dict = {}
        task_queue = list(task_dirs)
        finished = 0

        def _submit(td: str) -> None:
            slot = available_slots.pop(0)
            offset = port_base_offset + slot * _STRIDE
            fut = pool.submit(
                _run_single_decomposer_task,
                task_dir=td,
                config_path=args.config,
                decomposer_model=args.decomposer_model or args.model,
                executor_model=args.executor_model,
                api_key=args.api_key,
                base_url=args.base_url,
                trace_dir=batch_trace_dir,
                port_offset=offset,
                no_judge=args.no_judge,
                judge_model=getattr(args, "judge_model", None),
                trials=trials,
                proxy=getattr(args, "proxy", None),
                sandbox=getattr(args, "sandbox", False),
                sandbox_image=getattr(args, "sandbox_image", None),
                sandbox_tools=getattr(args, "sandbox_tools", False),
                no_vllm_wait=getattr(args, "no_vllm_wait", False),
                skip_grade=getattr(args, "skip_grade", False),
                no_sandbox=getattr(args, "no_sandbox", False),
            )
            pending[fut] = (td, slot)

        while task_queue and available_slots:
            _submit(task_queue.pop(0))

        while pending:
            for fut in as_completed(pending):
                td, slot = pending.pop(fut)
                available_slots.append(slot)
                finished += 1
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"task_id": Path(td).name, "error": str(e), "trials": []}
                results.append(res)

                out_path = Path(batch_trace_dir) / "batch_results.json"
                with open(out_path, "w") as pf:
                    json.dump(results, pf, indent=2, ensure_ascii=False)

                if res.get("error"):
                    status = "ERR"
                elif res.get("grading_skipped"):
                    status = "trace-only"
                else:
                    status = f"score={res.get('avg_score', 0):.2f}"
                print(f"[{finished}/{len(task_dirs)}] {res.get('task_id', Path(td).name)}: {status}")

                while task_queue and available_slots:
                    _submit(task_queue.pop(0))

    elapsed = time.monotonic() - start_time
    out_path = Path(batch_trace_dir) / "batch_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_ok = sum(1 for r in results if not r.get("error"))
    print(f"\nDone in {elapsed:.0f}s. {n_ok}/{len(results)} tasks succeeded.")
    print(f"Results: {out_path}")


def add_decomposer_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add shared decomposer CLI arguments to a subparser."""
    parser.add_argument("--decomposer-model", default=None, help="Decomposer model ID")
    parser.add_argument("--executor-model", default=None, help="Executor model ID")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials")
    parser.add_argument("--trace-dir", default=None, help="Output directory for traces")
    parser.add_argument("--judge-model", default=None, help="Override judge model ID")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM judge")
    parser.add_argument("--skip-grade", action="store_true", help="Write traces only; do not grade during the run")
    parser.add_argument("--port-offset", type=int, default=0, help="Service port offset")
    parser.add_argument("--sandbox", action="store_true", help="Run sandbox tools in Docker")
    parser.add_argument("--no-sandbox", action="store_true", help="Disable sandbox even if enabled in config")
    parser.add_argument("--sandbox-image", default=None, help="Override sandbox Docker image")
    parser.add_argument("--sandbox-tools", action="store_true", help="Inject sandbox tools without Docker")
    parser.add_argument("--proxy", default=None, help="HTTP proxy for model/judge API traffic")
    parser.add_argument("--model", default=None, help="Alias for --decomposer-model")
    parser.add_argument("--api-key", default=None, help="API key for both models")
    parser.add_argument("--base-url", default=None, help="Base URL for both models")
    parser.add_argument("--no-vllm-wait", action="store_true", help="Skip vLLM /v1/models readiness polling")
    parser.add_argument("--launch-vllm", action="store_true", help="Start/reuse local vLLM servers before running")
    parser.add_argument("--stop-vllm-on-exit", action="store_true", help="Stop vLLM servers launched by this command on exit")
    parser.add_argument("--vllm-host", default="127.0.0.1", help="Host for launched vLLM servers")
    parser.add_argument("--vllm-log-dir", default="logs/vllm", help="Directory for launched vLLM logs and pid files")
    parser.add_argument("--vllm-max-model-len", default=None, help="Optional --max-model-len for launched vLLM")
    parser.add_argument("--vllm-gpu-memory-utilization", default=None, help="Optional --gpu-memory-utilization for launched vLLM")
    parser.add_argument("--vllm-extra-arg", action="append", default=[], help="Extra argument(s) passed to vLLM; repeat as needed")
    parser.add_argument("--decomposer-gpu", default="0", help="GPU id(s) for launched decomposer/manager server")
    parser.add_argument("--decomposer-port", type=int, default=8000, help="Port for launched decomposer/manager server")
    parser.add_argument("--executor-gpu", default="1", help="GPU id(s) for launched executor server")
    parser.add_argument("--executor-port", type=int, default=8001, help="Port for launched executor server")


def main_decomposer(argv: list[str] | None = None) -> None:
    """Standalone entry point for scripts/run_decomposer.py."""
    parser = argparse.ArgumentParser(prog="run_decomposer", description="Claw-Eval decomposer runner")
    parser.add_argument("--task", required=True, help="Path to task dir or YAML")
    add_decomposer_cli_args(parser)
    args = parser.parse_args(argv)
    cmd_run_decomposer(args)
