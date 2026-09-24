import json
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from gyms.toolathlon_gym import run, usage
from sft.toolathlon_gym import collection as batch
from gyms.toolathlon_gym.subagents.model_config import generation_config, teacher_generation_config


def test_qwen_settings_preserve_upstream_profile() -> None:
    settings = generation_config("Qwen/Qwen3.5-4B")
    assert settings["temperature"] == 0.7
    assert settings["top_p"] == 0.8
    assert settings["presence_penalty"] == 1.5
    assert settings["extra_body"]["top_k"] == 20
    assert "reasoning_effort" not in settings["extra_body"]
    assert settings["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_flash_next_teacher_sends_explicit_low_reasoning():
    from decomposer.chat_vllm import ChatVLLM
    settings = teacher_generation_config("Qwen/Qwen3.8-Flash-Next-NVFP4")
    model = ChatVLLM(model="Qwen/Qwen3.8-Flash-Next-NVFP4",
                    api_key="test", base_url="https://router.test/v1", **settings)
    payload = model._get_request_payload("test")
    assert payload["reasoning_effort"] == "low"
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == .95
    assert payload["presence_penalty"] == 0.0
    assert payload["extra_body"] == {
        "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
        "include_reasoning": True, "chat_template_kwargs": {"enable_thinking": True},
    }
    assert model.preserve_reasoning is True
    assert "max_tokens" not in payload


def test_gemma_settings_preserve_collection_profile() -> None:
    settings = generation_config("google/gemma-4-26B-A4B-it")
    assert settings["temperature"] == 1.0
    assert settings["top_p"] == 0.95
    assert settings["extra_body"]["top_k"] == 64
    assert settings["extra_body"]["reasoning_effort"] == "none"
    assert settings["preserve_reasoning"] is False


def test_configured_subagents_are_registered() -> None:
    registered = json.loads(
        (Path(run.__file__).parent / "subagents" / "langgraph.json").read_text()
    )["graphs"]

    assert {
        assistant_id for _, assistant_id, _ in run.SUBAGENT_TYPES
    } <= registered.keys()
    assert [item[0] for item in run.SUBAGENT_TYPES] == [
        "gemma_4_26b_a4b_non_thinking"
    ]


def test_docker(monkeypatch) -> None:
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "output", "")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    result = run._docker("ps", check=False)

    assert result.stdout == "output"
    assert calls == [
        ((["docker", "ps"],), {"capture_output": True, "text": True})
    ]


def test_docker_error_includes_container_runtime_output(monkeypatch) -> None:
    monkeypatch.setattr(
        run.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 125, "", "podman socket is unavailable"
        ),
    )

    with pytest.raises(RuntimeError, match="podman socket is unavailable"):
        run._docker("ps")


def test_vllm_command_uses_current_environment_and_gemma_parsers() -> None:
    command = run.vllm_command(
        "/models/gemma",
        8023,
        max_model_len=32768,
        gpu_memory_utilization=0.8,
    )

    assert command[0] == str(Path(sys.executable).with_name("vllm"))
    assert command[1:3] == ["serve", "/models/gemma"]
    assert command[command.index("--served-model-name") + 1] == (
        run.DEFAULT_SUBAGENT_MODEL
    )
    assert command[command.index("--tool-call-parser") + 1] == "gemma4"
    assert command[command.index("--reasoning-parser") + 1] == "gemma4"
    assert command[command.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking":false}'
    )

    data_parallel_command = run.vllm_command(
        "/models/qwen",
        8023,
        max_model_len=32768,
        gpu_memory_utilization=0.8,
        data_parallel_size=4,
    )
    assert data_parallel_command[
        data_parallel_command.index("--data-parallel-size") + 1
    ] == "4"
    assert data_parallel_command[
        data_parallel_command.index("--api-server-count") + 1
    ] == "1"


def test_postgres_image_is_fully_qualified_for_podman() -> None:
    assert run.POSTGRES_IMAGE == "docker.io/library/postgres:15"


def test_start_vllm_adds_virtualenv_tools_to_path(tmp_path, monkeypatch) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    captured = {}
    process = SimpleNamespace()

    def fake_popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return process

    monkeypatch.setattr(run.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run, "wait_for_vllm", lambda *args, **kwargs: None)
    monkeypatch.setenv("PATH", "/usr/bin")

    assert run.start_vllm(
        model="model",
        port=port,
        gpu="0",
        max_model_len=1024,
        gpu_memory_utilization=0.5,
        timeout=1,
        log_path=tmp_path / "vllm.log",
        reuse=False,
    ) is process
    assert captured["env"]["PATH"].split(run.os.pathsep)[0] == str(
        Path(sys.executable).parent
    )


def test_postgres_environment_uses_container_ip_instead_of_dns(monkeypatch) -> None:
    monkeypatch.setattr(
        run,
        "_docker",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, "10.89.3.7\n", ""
        ),
    )

    environment = run._postgres_environment("postgres-container")

    assert environment["PGHOST"] == "10.89.3.7"
    assert environment["PG_HOST"] == "10.89.3.7"
    assert environment["PGDATABASE"] == "toolathlon_gym"


def test_raw_runner_has_no_collection_policy() -> None:
    args = run.create_parser().parse_args(["example", "--harness", "react"])
    assert args.purpose == "raw"
    assert args.harness == "react"
    assert not hasattr(args, "adaptive")


def test_main_rejects_path_traversal(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "../finalpool", "--purpose", "trace-generation"],
    )

    with pytest.raises(ValueError, match="Tasks must be unique"):
        run.main()


def test_task_selection_ignores_helpers_and_validates_subset(tmp_path) -> None:
    (tmp_path / "beta").mkdir()
    (tmp_path / ".utils").mkdir()
    (tmp_path / "alpha").mkdir()

    assert batch.select_tasks(tmp_path, run_all=True, requested=None) == [
        "alpha",
        "beta",
    ]
    assert batch.select_tasks(
        tmp_path, run_all=False, requested=["beta", "alpha"]
    ) == ["beta", "alpha"]
    with pytest.raises(ValueError, match="Unknown Toolathlon task"):
        batch.select_tasks(tmp_path, run_all=False, requested=["missing"])

    assert batch.wants_batch(["--repetitions=2"])
    assert batch.wants_batch(["-n2"])


def test_batch_uses_uncapped_teacher_and_extended_timeouts(monkeypatch) -> None:
    monkeypatch.delenv("DECOMPOSER_MAX_TOKENS", raising=False)
    args = batch.parse_args(
        ["--all", "--purpose", "trace-generation"],
        {
            "model": "teacher",
            "subagent_model": "student",
            "subagent_port": 8031,
            "image": "image",
            "artifacts_dir": Path("artifacts"),
        },
    )

    assert args.agent_timeout == 2700
    assert args.episode_timeout == 3300


def test_usage_summary_separates_teacher_and_subagents() -> None:
    def message(input_tokens, output_tokens, *, cache=0, reasoning=0):
        return {
            "type": "ai",
            "data": {
                "usage_metadata": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "input_token_details": {"cache_read": cache},
                    "output_token_details": {"reasoning": reasoning},
                }
            },
        }

    summary = usage.build_usage_summary(
        [message(100, 10, cache=40, reasoning=3)],
        {
            "sub-1": {
                "subagent_id": "worker-1",
                "status": "responded",
                "messages": [message(20, 5)],
            }
        },
        {"worker-1": {"subagent_type_id": "qwen"}},
    )

    assert summary["subagents"]["sub-1"]["subagent_type_id"] == "qwen"
    assert summary["decomposer"]["total_tokens"] == 110
    assert summary["subagents"]["sub-1"]["total_tokens"] == 25
    assert summary["totals"]["total_tokens"] == 135
    assert summary["totals"]["cache_read_tokens"] == 40
    assert summary["totals"]["reasoning_tokens"] == 3


def test_teacher_credentials_accept_lmrouter(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROXY_URL", "https://lmrouter.example/v1")
    monkeypatch.setenv("LLM_PROXY_MASTER_KEY", "test-key")

    batch.validate_teacher_credentials()


def test_teacher_credentials_accept_local_vllm(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROXY_URL", raising=False)
    monkeypatch.delenv("LLM_PROXY_MASTER_KEY", raising=False)
    monkeypatch.setenv("DECOMPOSER_VLLM_BASE_URL", "http://127.0.0.1:8031/v1")

    batch.validate_teacher_credentials()


def test_teacher_credentials_require_complete_lmrouter_config(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROXY_URL", "https://lmrouter.example/v1")
    monkeypatch.delenv("LLM_PROXY_MASTER_KEY", raising=False)

    with pytest.raises(RuntimeError, match="both LLM_PROXY_URL"):
        batch.validate_teacher_credentials()


def test_batch_repetitions_and_resume_skip_completed(tmp_path, monkeypatch) -> None:
    toolathlon_root = tmp_path / "toolathlon"
    for task in ("alpha", "beta"):
        (toolathlon_root / "tasks" / "finalpool" / task).mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(batch, "new_run_id", lambda: "test-run")

    vllm_starts = []
    vllm_stops = []
    episode_calls = []

    def fake_start_vllm(**kwargs):
        vllm_starts.append(kwargs)
        return "process"

    def fake_execute_episode(args, **kwargs):
        episode_calls.append(kwargs)
        if len(episode_calls) == 1:
            manifest_path = artifacts / "runs/test-run/manifest.json"
            before = manifest_path.read_bytes()
            with pytest.raises(RuntimeError, match="already active"):
                batch.main([
                    "--resume", "test-run", "--purpose", "trace-generation",
                    "--gym-artifacts-dir", str(artifacts),
                ], **common)
            assert manifest_path.read_bytes() == before
        task = kwargs["episode"]["task"]
        repetition = kwargs["episode"]["repetition"]
        attempt = kwargs["attempt"]
        return {
            "attempt": attempt,
            "status": "completed",
            "score": False,
            "artifact_path": f"/traces/{task}/{repetition}/{attempt}",
            "evaluation_path": f"/evals/{task}/{repetition}/{attempt}/result.json",
            "started_at": "start",
            "finished_at": "finish",
            "duration_seconds": 1.25,
            "returncode": 0,
            "error": None,
        }

    monkeypatch.setattr(batch, "execute_episode", fake_execute_episode)
    common = {
        "repo_root": tmp_path,
        "toolathlon_root": toolathlon_root,
        "default_artifacts_dir": artifacts,
        "default_image": "image",
        "default_model": "decomposer-model",
        "default_subagent_model": "subagent-model",
        "default_subagent_port": 8023,
        "start_vllm": fake_start_vllm,
        "stop_vllm": vllm_stops.append,
        "docker": lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    }

    manifest = batch.main(
        [
            "--tasks", "alpha", "beta", "-n", "2",
            "--purpose", "trace-generation",
            "--gym-artifacts-dir", str(artifacts),
        ],
        **common,
    )

    assert manifest["status"] == "completed"
    assert manifest["counts"] == {
        "pending": 0,
        "running": 0,
        "completed": 4,
        "failed": 0,
        "total": 4,
    }
    assert len(episode_calls) == 4
    assert len(vllm_starts) == 1
    assert vllm_stops == ["process"]
    assert all(episode["score"] is False for episode in manifest["episodes"])

    manifest["episodes"][1]["status"] = "failed"
    batch.save_manifest(artifacts / "runs" / "test-run", manifest)
    episode_calls.clear()
    resumed = batch.main(
        [
            "--resume", "test-run", "--purpose", "trace-generation",
            "--gym-artifacts-dir", str(artifacts),
        ],
        **common,
    )

    assert len(episode_calls) == 1
    assert episode_calls[0]["episode"]["task"] == "alpha"
    assert episode_calls[0]["episode"]["repetition"] == 2
    assert episode_calls[0]["attempt"] == 2
    assert resumed["counts"]["completed"] == 4
    assert len(resumed["episodes"][1]["attempts"]) == 2
    assert len(vllm_starts) == 2


@pytest.mark.parametrize("hosted", [False, True])
def test_batch_runs_episodes_with_requested_concurrency(tmp_path, monkeypatch, hosted) -> None:
    toolathlon_root = tmp_path / "toolathlon"
    tasks = [f"task-{index}" for index in range(4)]
    for task in tasks:
        (toolathlon_root / "tasks" / "finalpool" / task).mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(batch, "new_run_id", lambda: "concurrent-run")

    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_execute_episode(args, **kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        attempt = kwargs["attempt"]
        return {
            "attempt": attempt,
            "status": "completed",
            "score": True,
            "artifact_path": "/trace",
            "evaluation_path": "/eval/result.json",
            "started_at": "start",
            "finished_at": "finish",
            "duration_seconds": 0.05,
            "returncode": 0,
            "error": None,
        }

    monkeypatch.setattr(batch, "execute_episode", fake_execute_episode)
    starts = []
    manifest = batch.main(
        [
            "--tasks",
            *tasks,
            "--purpose",
            "trace-generation",
            "--concurrency",
            "4",
            "--gym-artifacts-dir",
            str(artifacts),
            *(["--subagent-base-url", "https://router.test/v1"] if hosted else []),
        ],
        repo_root=tmp_path,
        toolathlon_root=toolathlon_root,
        default_artifacts_dir=artifacts,
        default_image="image",
        default_model="decomposer-model",
        default_subagent_model="subagent-model",
        default_subagent_port=8023,
        start_vllm=lambda **kwargs: starts.append(kwargs) or "process",
        stop_vllm=lambda process: None,
        docker=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )

    assert maximum_active == 4
    assert len(starts) == (0 if hosted else 1)
    assert manifest["config"]["subagent_base_url"] == ("https://router.test/v1" if hosted else None)
    assert manifest["counts"]["completed"] == 4


def test_provider_backoff_coalesces_failures_and_escalates_by_generation() -> None:
    now = [100.0]
    barrier = batch.ProviderBackoffBarrier(
        initial_seconds=30, maximum_seconds=100, clock=lambda: now[0]
    )

    assert barrier.generation() == 0
    assert barrier.register_failure(0) == 30
    assert barrier.register_failure(0) is None
    now[0] = 130.0
    assert barrier.wait() == 1
    assert barrier.register_failure(1) == 60
    assert barrier.register_failure(1) is None
    now[0] = 190.0
    assert barrier.wait() == 2
    assert barrier.register_failure(2) == 100


def test_provider_backoff_resets_after_current_generation_success() -> None:
    now = [0.0]
    barrier = batch.ProviderBackoffBarrier(clock=lambda: now[0])

    assert barrier.register_failure(0) == 30
    assert barrier.register_success(0) is False
    now[0] = 30.0
    generation = barrier.wait()
    assert barrier.register_success(generation) is True
    assert barrier.register_failure(generation) == 30


def test_openrouter_transient_failure_uses_saved_decomposer_error(tmp_path) -> None:
    artifact = tmp_path / "trace"
    artifact.mkdir()
    (artifact / "trace.json").write_text(
        json.dumps({"agent_error": "WriteTimeout('timed out')"})
    )
    result = {
        "status": "failed",
        "artifact_path": str(artifact),
        "error": {"stderr_tail": "Decomposer agent loop failed"},
    }

    assert batch.openrouter_transient_failure(result) == "writetimeout"


def test_openrouter_transient_failure_ignores_unrelated_local_timeout() -> None:
    result = {
        "status": "failed",
        "artifact_path": None,
        "error": {"stderr_tail": "local subagent raised httpx.WriteTimeout"},
    }

    assert batch.openrouter_transient_failure(result) is None


def test_batch_distributes_episodes_across_external_vllm_ports(
    tmp_path, monkeypatch
) -> None:
    toolathlon_root = tmp_path / "toolathlon"
    tasks = [f"task-{index}" for index in range(6)]
    for task in tasks:
        (toolathlon_root / "tasks" / "finalpool" / task).mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(batch, "new_run_id", lambda: "endpoint-pool-run")

    ports = []

    def fake_execute_episode(args, **kwargs):
        ports.append(kwargs["subagent_port"])
        attempt = kwargs["attempt"]
        return {
            "attempt": attempt,
            "status": "completed",
            "score": True,
            "artifact_path": "/trace",
            "evaluation_path": "/eval/result.json",
            "started_at": "start",
            "finished_at": "finish",
            "duration_seconds": 0.01,
            "returncode": 0,
            "error": None,
        }

    monkeypatch.setattr(batch, "execute_episode", fake_execute_episode)
    starts = []
    manifest = batch.main(
        [
            "--tasks",
            *tasks,
            "--purpose",
            "trace-generation",
            "--subagent-ports",
            "18100",
            "18101",
            "18102",
            "--concurrency",
            "3",
            "--gym-artifacts-dir",
            str(artifacts),
        ],
        repo_root=tmp_path,
        toolathlon_root=toolathlon_root,
        default_artifacts_dir=artifacts,
        default_image="image",
        default_model="decomposer-model",
        default_subagent_model="subagent-model",
        default_subagent_port=8023,
        start_vllm=lambda **kwargs: starts.append(kwargs) or None,
        stop_vllm=lambda process: None,
        docker=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )

    assert [item["port"] for item in starts] == [18100, 18101, 18102]
    assert all(item["reuse"] for item in starts)
    assert Counter(ports) == Counter({18100: 2, 18101: 2, 18102: 2})
    assert manifest["config"]["subagent_ports"] == [18100, 18101, 18102]


def test_cleanup_continues_when_log_capture_fails(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_docker(*args, check=True):
        calls.append(args)
        if args[0] == "logs":
            raise OSError("capture failed")
        return subprocess.CompletedProcess(args, 0, "inspect", "")

    monkeypatch.setattr(run, "_docker", fake_docker)
    run._cleanup_episode(
        episode_dir=tmp_path,
        task_container="task-container",
        pg_container="pg-container",
        network="episode-network",
    )

    assert ("rm", "--force", "--volumes", "task-container") in calls
    assert ("rm", "--force", "--volumes", "pg-container") in calls
    assert ("network", "rm", "episode-network") in calls
    cleanup = json.loads((tmp_path / "cleanup.json").read_text())
    assert cleanup["captures"][0]["error"] == "OSError('capture failed')"


def test_interrupted_attempt_is_recorded_for_next_resume(tmp_path, monkeypatch) -> None:
    toolathlon_root = tmp_path / "toolathlon"
    (toolathlon_root / "tasks" / "finalpool" / "alpha").mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(batch, "new_run_id", lambda: "interrupted-run")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(batch, "execute_episode", interrupt)

    with pytest.raises(KeyboardInterrupt):
        batch.main(
            [
                "--tasks", "alpha", "--purpose", "trace-generation",
                "--gym-artifacts-dir", str(artifacts),
            ],
            repo_root=tmp_path,
            toolathlon_root=toolathlon_root,
            default_artifacts_dir=artifacts,
            default_image="image",
            default_model="decomposer-model",
            default_subagent_model="subagent-model",
            default_subagent_port=8023,
            start_vllm=lambda **kwargs: "process",
            stop_vllm=lambda process: None,
            docker=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
        )

    manifest = batch.load_manifest(artifacts / "runs" / "interrupted-run")
    assert manifest["status"] == "interrupted"
    assert manifest["episodes"][0]["status"] == "failed"
    assert manifest["episodes"][0]["attempts"][0]["attempt"] == 1
    assert manifest["episodes"][0]["attempts"][0]["error"]["interrupted"] is True


def test_reconcile_preserves_attempt_left_by_abrupt_exit(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-id"
    orphan = run_dir / "attempts" / "alpha" / "rep-001" / "attempt-001"
    orphan.mkdir(parents=True)
    episode = {
        "key": "alpha::rep-001",
        "task": "alpha",
        "repetition": 1,
        "status": "running",
        "attempts": [],
    }

    attempt, changed = batch.next_attempt(run_dir, episode)

    assert changed is True
    assert attempt == 2
    assert episode["status"] == "failed"
    assert episode["attempts"][0]["attempt"] == 1
    assert episode["attempts"][0]["attempt_log_path"] == str(orphan)
    assert episode["attempts"][0]["error"]["type"] == "RecoveredIncompleteAttempt"


def test_resume_run_id_must_be_one_path_component() -> None:
    with pytest.raises(ValueError, match="Invalid run ID"):
        batch.validate_run_id("../outside")


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_recovery_reads_saved_attempt_before_retrying(tmp_path, status):
    directory = tmp_path / "attempts/alpha/rep-001/attempt-001"
    directory.mkdir(parents=True)
    saved = {"attempt": 1, "status": status, "score": status == "completed",
             "evaluation_path": "/saved/result.json", "error": None}
    (directory / "attempt.json").write_text(json.dumps(saved))
    episode = {"task": "alpha", "repetition": 1, "status": "running", "attempts": []}
    assert batch.next_attempt(tmp_path, episode) == (2, True)
    assert episode["attempts"] == [saved]
    assert episode["status"] == status
    assert episode["evaluation_path"] == saved["evaluation_path"]
    assert batch.next_attempt(tmp_path, episode) == (2, False)


def test_run_lock_released_on_exception(tmp_path):
    with pytest.raises(ValueError):
        with batch.run_lock(tmp_path):
            raise ValueError("crash")
    with batch.run_lock(tmp_path):
        pass


def test_cleanup_inspection_never_saves_container_secrets(tmp_path, monkeypatch):
    secret = "test-secret-do-not-export"
    inspect = [{"Id": "id", "Name": "task", "Image": "sha256:test",
                "State": {"Status": "exited", "ExitCode": 1, "Error": secret},
                "Config": {"Env": [f"VLLM_API_KEY={secret}"]}, "Args": [secret]}]
    def docker(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0,
            json.dumps(inspect) if args[0] == "inspect" else "", "")
    monkeypatch.setattr(run, "_docker", docker)
    run._cleanup_episode(episode_dir=tmp_path, task_container="task",
                         pg_container="postgres", network="net")
    for path in tmp_path.iterdir():
        assert secret not in path.read_text()
    state = json.loads((tmp_path / "task.inspect.json").read_text())[0]
    assert state["State"]["ExitCode"] == 1
    assert "Config" not in state


@pytest.mark.parametrize("error", [None, TimeoutError("deadline"), RuntimeError("model failed")])
def test_agent_failure_is_evaluated_before_cleanup(tmp_path, monkeypatch, error):
    from contextlib import nullcontext
    root = tmp_path / "gym"
    (root / "tasks/finalpool/alpha").mkdir(parents=True)
    monkeypatch.setattr(run, "TOOLATHLON_ROOT", root)
    monkeypatch.setenv("VLLM_API_KEY", "fixture")
    monkeypatch.delenv("LLM_PROXY_UNIX_SOCKET", raising=False)
    args = run.create_parser().parse_args([
        "alpha", "--harness", "react", "--episode-id", "fixture",
        "--subagent-base-url", "https://router.test/v1",
        "--artifacts-dir", str(tmp_path / "traces"),
        "--evals-dir", str(tmp_path / "evals"),
    ])
    evaluated = []
    result_path = tmp_path / "evals/alpha/fixture/result.json"
    def docker(*command, **kwargs):
        output = ""
        if command[0] == "inspect":
            output = "healthy" if "--format" in command else "[]"
        elif "/proc/1/comm" in command:
            output = "postgres"
        elif "--tuples-only" in command:
            output = "t"
        elif command[0] == "port":
            output = "127.0.0.1:12345"
        elif command[0] == "run" and "127.0.0.1::2024" in command:
            runtime = {"task_config": {"task_str": "fixture", "agent_workspace": "/work",
                "launch_time": None, "evaluation": {"evaluation_command": "evaluate",
                                                     "groundtruth_workspace": None}}}
            (tmp_path / "traces/alpha/fixture/runtime.json").write_text(json.dumps(runtime))
        elif "--res_log_file" in command:
            evaluated.append(True)
        elif "/tmp/decomposer-evaluation.json" in command:
            output = '{"passed": 10, "total": 10}'
        elif command[0] == "rm":
            assert result_path.is_file()
        return subprocess.CompletedProcess(command, 0, output, "")
    monkeypatch.setattr(run, "_docker", docker)
    monkeypatch.setattr(run, "_postgres_environment", lambda container: {})
    monkeypatch.setattr(run.urllib.request, "urlopen",
                        lambda *a, **kw: nullcontext(SimpleNamespace(status=200)))
    monkeypatch.setattr(run, "make_agent", lambda *a: (None, {"configurable": {"thread_id": "t"}}))
    async def invoke(*a):
        return {"messages": []}, error
    monkeypatch.setattr(run, "invoke_and_capture", invoke)
    with pytest.raises(RuntimeError, match="Agent loop failed") if error else nullcontext():
        run.run_episode(args)
    assert evaluated == [True]
    result = json.loads(result_path.read_text())
    assert result["native_pass"] is True
    assert result["pass"] is (error is None)
    assert result["agent_error"] == (repr(error) if error else None)


def test_execute_episode_maps_deterministic_trace_and_eval_paths(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "artifacts"
    run_dir = root / "runs" / "run-id"
    episode_id = batch.episode_id_for("run-id", "alpha", 2, 1)
    artifact_dir = root / "traces" / "alpha" / episode_id
    evaluation_path = root / "evals" / "alpha" / episode_id / "result.json"
    artifact_dir.mkdir(parents=True)
    evaluation_path.parent.mkdir(parents=True)
    evaluation_path.write_text('{"pass": false}')

    class FakeProcess:
        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    popen_calls = []

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(batch.subprocess, "Popen", fake_popen)
    args = SimpleNamespace(
        purpose="trace-generation",
        model="decomposer-model",
        subagent_model="subagent-model",
        subagent_port=8023,
        subagent_gpu="1",
        vllm_max_model_len=32768,
        vllm_gpu_memory_utilization=0.8,
        vllm_startup_timeout=30,
        image="image",
        startup_timeout=10,
        n_jobs_per_worker=1000,
    )

    result = batch.execute_episode(
        args,
        runner_path=tmp_path / "run.py",
        root=root,
        run_dir=run_dir,
        episode={"task": "alpha", "repetition": 2},
        attempt=1,
    )

    assert result["status"] == "completed"
    assert result["score"] is False
    assert result["artifact_path"] == str(artifact_dir)
    assert result["evaluation_path"] == str(evaluation_path)
    command = popen_calls[0][0]
    assert command[command.index("--episode-id") + 1] == episode_id
    assert command[command.index("--run-id") + 1] == "run-id"
    assert command[command.index("--n-jobs-per-worker") + 1] == "1000"
    assert command[command.index("--container-lock-file") + 1] == str(
        root / "runs" / "run-id" / "container.lock"
    )


def test_episode_command_distributes_container_operations_across_slots(tmp_path) -> None:
    args = SimpleNamespace(
        purpose="trace-generation",
        model="decomposer-model",
        subagent_model="subagent-model",
        subagent_port=8023,
        subagent_gpu="0,1",
        vllm_max_model_len=32768,
        vllm_gpu_memory_utilization=0.8,
        vllm_data_parallel_size=2,
        vllm_startup_timeout=30,
        image="image",
        startup_timeout=10,
        n_jobs_per_worker=1000,
        agent_timeout=1200,
        container_slots=2,
        subagent_base_url="https://router.test/v1",
        subagent_host="router.test:192.0.2.1",
    )

    command = batch.episode_command(
        args,
        tmp_path / "run.py",
        "alpha",
        "run-id",
        1,
        1,
        "episode-id",
        tmp_path,
        container_slot=1,
    )

    assert command[command.index("--container-lock-file") + 1] == str(
        tmp_path / "runs" / "run-id" / "container-01.lock"
    )
    assert command[command.index("--vllm-data-parallel-size") + 1] == "2"
    assert command[command.index("--agent-timeout") + 1] == "1200"
    assert command[command.index("--subagent-base-url") + 1] == "https://router.test/v1"
    assert command[command.index("--subagent-host") + 1] == "router.test:192.0.2.1"
