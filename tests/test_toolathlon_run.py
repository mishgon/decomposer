import json
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from gyms.toolathlon import batch, mlspace_serve, run, settings


def test_evaluation_runtime_settings_are_uniform() -> None:
    assert settings.SUBAGENT_CONTEXT_TOKENS == 265_000
    assert settings.SUBAGENT_RECURSION_LIMIT == 410
    assert settings.DECOMPOSER_RECURSION_LIMIT == 410
    assert settings.DEEPSEEK_REASONING_EFFORT == "high"


def test_configured_subagents_are_registered() -> None:
    registered = json.loads(
        (Path(run.__file__).parent / "subagents" / "langgraph.json").read_text()
    )["graphs"]

    assert {
        assistant_id for _, assistant_id, _ in run.SUBAGENT_TYPES
    } <= registered.keys()
    assert [item[0] for item in run.SUBAGENT_TYPES] == [
        "qwen_3_5_4b_non_thinking"
    ]


def test_docker(monkeypatch) -> None:
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "output", "")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    result = run._docker("ps", check=False)

    assert result.stdout == "output"
    assert calls == [((["docker", "ps"],), {"capture_output": True, "text": True})]


def test_docker_failure_includes_stderr(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args, 125, "", "Error: statfs /x: no such file"
        )

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    with pytest.raises(
        RuntimeError,
        match="docker run x failed with exit code 125: "
        "Error: statfs /x: no such file",
    ):
        run._docker("run", "x")


def test_vllm_command_uses_qwen_parsers() -> None:
    command = run.vllm_command(
        "/models/qwen",
        8030,
        max_model_len=32768,
        gpu_memory_utilization=0.8,
    )

    assert command[0] == str(Path(sys.executable).with_name("vllm"))
    assert command[1:3] == ["serve", "/models/qwen"]
    assert command[command.index("--served-model-name") + 1] == (
        run.DEFAULT_SUBAGENT_MODEL
    )
    assert command[command.index("--tool-call-parser") + 1] == "qwen3_xml"
    assert command[command.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking":false}'
    )


def test_main_requires_explicit_purpose(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run.py", "finalpool/example"])

    with pytest.raises(SystemExit):
        run.main()


def test_main_rejects_unknown_purpose(monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["run.py", "finalpool/example", "--purpose", "sft"]
    )

    with pytest.raises(SystemExit):
        run.main()


def test_main_accepts_evaluation_purpose(monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["run.py", "finalpool/example", "--purpose", "evaluation"]
    )

    with pytest.raises(ValueError, match="Unknown Toolathlon task"):
        run.main()


def test_main_fails_fast_when_no_docker_socket(tmp_path, monkeypatch) -> None:
    (tmp_path / "tasks" / "finalpool" / "example").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "global_configs.py").write_text("# test\n")
    (tmp_path / "configs" / "token_key_session_example.py").write_text(
        "tokens = {}\n"
    )
    monkeypatch.setattr(run, "TOOLATHLON_ROOT", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(run, "DOCKER_SOCKET_CANDIDATES", (str(tmp_path / "nope"),))
    monkeypatch.setattr(
        sys, "argv", ["run.py", "finalpool/example", "--purpose", "evaluation"]
    )

    with pytest.raises(RuntimeError, match="No Docker socket found"):
        run.main()


def test_main_bootstraps_global_configs_from_example(tmp_path, monkeypatch) -> None:
    (tmp_path / "tasks" / "finalpool" / "example").mkdir(parents=True)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "global_configs_example.py").write_text("global_configs = {}\n")
    (configs / "token_key_session_example.py").write_text("tokens = {}\n")
    monkeypatch.setattr(run, "TOOLATHLON_ROOT", tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["run.py", "finalpool/example", "--purpose", "evaluation"]
    )

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        run.main()

    assert (configs / "global_configs.py").read_text() == "global_configs = {}\n"
    assert (configs / "token_key_session.py").read_text() == "tokens = {}\n"
    assert (configs / ".mcp-auth").is_dir()


def test_main_fails_when_checkout_incomplete(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run, "TOOLATHLON_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["run.py", "finalpool/example", "--purpose", "evaluation"]
    )

    with pytest.raises(RuntimeError, match="git submodule update --init"):
        run.main()


def test_main_fails_when_global_configs_unseedable(tmp_path, monkeypatch) -> None:
    (tmp_path / "tasks" / "finalpool" / "example").mkdir(parents=True)
    monkeypatch.setattr(run, "TOOLATHLON_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["run.py", "finalpool/example", "--purpose", "evaluation"]
    )

    with pytest.raises(RuntimeError, match="global_configs_example"):
        run.main()


def test_batch_fails_when_checkout_incomplete(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(RuntimeError, match="git submodule update --init"):
        batch.main(
            [
                "--all",
                "--purpose",
                "evaluation",
                "--bench-artifacts-dir",
                str(tmp_path / "artifacts"),
            ],
            repo_root=tmp_path,
            toolathlon_root=tmp_path / "missing",
            default_artifacts_dir=tmp_path / "artifacts",
            default_image="image",
            default_model="decomposer-model",
            default_subagent_model="subagent-model",
            default_subagent_port=8030,
            start_vllm=lambda **kwargs: None,
            stop_vllm=lambda process: None,
            docker=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
        )


def test_resolve_docker_socket_explicit_wins(tmp_path, monkeypatch) -> None:
    socket_path = tmp_path / "custom.sock"
    socket_path.touch()
    monkeypatch.setenv("DOCKER_HOST", "unix:///nonexistent")

    assert run.resolve_docker_socket(str(socket_path)) == str(socket_path)


def test_resolve_docker_socket_accepts_absolute_daemon_side_path() -> None:
    assert run.resolve_docker_socket("/var/run/docker.sock") == (
        "/var/run/docker.sock"
    )


def test_resolve_docker_socket_rejects_relative_explicit() -> None:
    with pytest.raises(RuntimeError, match="must be absolute"):
        run.resolve_docker_socket("relative/docker.sock")


def test_resolve_docker_socket_uses_docker_host_unix(tmp_path, monkeypatch) -> None:
    socket_path = tmp_path / "host.sock"
    socket_path.touch()
    monkeypatch.setenv("DOCKER_HOST", f"unix://{socket_path}")
    monkeypatch.setattr(run, "DOCKER_SOCKET_CANDIDATES", ())

    assert run.resolve_docker_socket() == str(socket_path)


def test_resolve_docker_socket_falls_back_to_candidates(tmp_path, monkeypatch) -> None:
    first = tmp_path / "var.sock"
    second = tmp_path / "run.sock"
    second.touch()
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(run, "DOCKER_SOCKET_CANDIDATES", (str(first), str(second)))

    assert run.resolve_docker_socket() == str(second)


def test_resolve_docker_socket_uses_rootless_xdg_socket(tmp_path, monkeypatch) -> None:
    socket_dir = tmp_path / "xdg"
    socket_dir.mkdir()
    socket_path = socket_dir / "docker.sock"
    socket_path.touch()
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(socket_dir))
    monkeypatch.setattr(run, "DOCKER_SOCKET_CANDIDATES", ())

    assert run.resolve_docker_socket() == str(socket_path)


def test_resolve_docker_socket_uses_rootless_podman_socket(tmp_path, monkeypatch) -> None:
    socket_dir = tmp_path / "xdg" / "podman"
    socket_dir.mkdir(parents=True)
    socket_path = socket_dir / "podman.sock"
    socket_path.touch()
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "xdg"))
    monkeypatch.setattr(run, "DOCKER_SOCKET_CANDIDATES", ())

    assert run.resolve_docker_socket() == str(socket_path)


def test_resolve_docker_socket_fails_when_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(
        run, "DOCKER_SOCKET_CANDIDATES", (str(tmp_path / "missing.sock"),)
    )

    with pytest.raises(RuntimeError, match="No Docker socket found"):
        run.resolve_docker_socket()


def test_episode_command_passes_docker_socket(tmp_path) -> None:
    args = SimpleNamespace(
        purpose="evaluation",
        model="model",
        subagent_model="subagent-model",
        subagent_port=8030,
        subagent_base_url="http://host.docker.internal:8030/v1",
        subagent_gpu="0",
        vllm_max_model_len=65536,
        subagent_recursion_limit=410,
        vllm_gpu_memory_utilization=0.9,
        vllm_startup_timeout=1800,
        image="image",
        docker_socket="/custom/docker.sock",
        startup_timeout=180,
        n_jobs_per_worker=1000,
        max_steps=200,
        eval_config="scripts/formal_run_v0.json",
    )

    command = batch.episode_command(
        args, Path("run.py"), "finalpool/example", "run-1", 1, 1, "ep-1", tmp_path
    )

    assert command[command.index("--docker-socket") + 1] == "/custom/docker.sock"
    assert command[command.index("--subagent-base-url") + 1] == (
        "http://host.docker.internal:8030/v1"
    )

    args.docker_socket = None
    command = batch.episode_command(
        args, Path("run.py"), "finalpool/example", "run-1", 1, 1, "ep-1", tmp_path
    )

    assert "--docker-socket" not in command


def test_main_rejects_path_traversal(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "../finalpool", "--purpose", "trace-generation"],
    )

    with pytest.raises(ValueError, match="Unknown Toolathlon task"):
        run.main()


def test_main_rejects_single_level_task(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "example-task", "--purpose", "trace-generation"],
    )

    with pytest.raises(ValueError, match="Unknown Toolathlon task"):
        run.main()


def test_task_selection_lists_domain_task_pairs_and_validates_subset(tmp_path) -> None:
    for task in ("alpha", "beta"):
        (tmp_path / "finalpool" / task).mkdir(parents=True)
    (tmp_path / "finalpool" / ".utils").mkdir()

    assert batch.select_tasks(tmp_path, run_all=True, requested=None) == [
        "finalpool/alpha",
        "finalpool/beta",
    ]
    assert batch.select_tasks(
        tmp_path, run_all=False, requested=["finalpool/beta", "finalpool/alpha"]
    ) == ["finalpool/beta", "finalpool/alpha"]
    with pytest.raises(ValueError, match="Unknown Toolathlon task"):
        batch.select_tasks(tmp_path, run_all=False, requested=["finalpool/missing"])

    assert batch.wants_batch(["--repetitions=2"])
    assert batch.wants_batch(["-n2"])


def test_validate_bundle_accepts_trusted_bundle_and_rejects_tampering(tmp_path) -> None:
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    bundle = {
        "schema_version": 2,
        "task_dir": "finalpool/alpha",
        "resolved_task_config": {
            "task_root": "/workspace/dumps",
            "agent_workspace": "/workspace/dumps/workspace",
            "log_file": "/workspace/dumps/traj_log.json",
        },
        "container_paths": {
            "task_root": "/workspace/dumps",
            "agent_workspace": "/workspace/dumps/workspace",
            "log_file": "/workspace/dumps/traj_log.json",
        },
        "host_paths": {
            "task_root": str(episode_dir.resolve()),
            "agent_workspace": str(episode_dir.resolve() / "workspace"),
            "log_file": str(episode_dir.resolve() / "traj_log.json"),
        },
    }

    run.validate_bundle(bundle, "finalpool/alpha", str(episode_dir.resolve()))

    with pytest.raises(ValueError, match="unsupported task bundle"):
        run.validate_bundle(
            {**bundle, "schema_version": 1}, "finalpool/alpha", str(episode_dir)
        )
    with pytest.raises(ValueError, match="task_dir mismatch"):
        run.validate_bundle(
            {**bundle, "task_dir": "finalpool/other"},
            "finalpool/alpha",
            str(episode_dir),
        )
    with pytest.raises(ValueError, match="below /workspace"):
        run.validate_bundle(
            {
                **bundle,
                "container_paths": {**bundle["container_paths"], "task_root": "/tmp/dumps"},
                "resolved_task_config": {**bundle["resolved_task_config"], "task_root": "/tmp/dumps"},
            },
            "finalpool/alpha",
            str(episode_dir),
        )
    with pytest.raises(ValueError, match="host output root mismatch"):
        run.validate_bundle(
            {**bundle, "host_paths": {**bundle["host_paths"], "task_root": str(tmp_path)}},
            "finalpool/alpha",
            str(episode_dir),
        )


def test_write_trajectory_matches_evaluator_contract(tmp_path) -> None:
    state = {
        "messages": [
            HumanMessage(content="do the task"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "spawn_subagent", "args": {"p": 1}, "id": "tc-1"}
                ],
            ),
            AIMessage(content="final answer"),
        ]
    }
    subagent_runs = {
        "r1": {
            "subagent_run_id": "r1",
            "tool_calls": [{"id": "x1", "name": "gw-search", "args": {"q": 1}}],
        }
    }

    run.write_trajectory(
        tmp_path,
        task="finalpool/alpha",
        episode_id="episode-1",
        status="success",
        state=state,
        subagent_runs=subagent_runs,
        started_at="2026-01-01T00:00:00Z",
        resolved_task_config={"task_root": "/workspace/dumps"},
    )

    trajectory = json.loads((tmp_path / "traj_log.json").read_text(encoding="utf-8"))
    assert trajectory["status"] == "success"
    assert trajectory["config"] == {"task_root": "/workspace/dumps"}
    assert trajectory["messages"][0] == {"role": "user", "content": "do the task"}
    assert trajectory["messages"][1]["role"] == "assistant"
    assert trajectory["messages"][1]["tool_calls"] == [
        {"id": "tc-1", "name": "spawn_subagent", "args": {"p": 1}}
    ]
    assert "spawn_subagent" in trajectory["tool_calls"]["tools"]
    assert "gw-search" in trajectory["tool_calls"]["tools"]
    assert trajectory["key_stats"]["subagent_runs"] == 1
    assert trajectory["session_id"] == "episode-1"


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
        "default_subagent_port": 8030,
        "start_vllm": fake_start_vllm,
        "stop_vllm": vllm_stops.append,
        "docker": lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    }

    manifest = batch.main(
        [
            "--tasks", "finalpool/alpha", "finalpool/beta", "-n", "2",
            "--purpose", "trace-generation",
            "--bench-artifacts-dir", str(artifacts),
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
            "--bench-artifacts-dir", str(artifacts),
        ],
        **common,
    )

    assert len(episode_calls) == 1
    assert episode_calls[0]["episode"]["task"] == "finalpool/alpha"
    assert episode_calls[0]["episode"]["repetition"] == 2
    assert episode_calls[0]["attempt"] == 2
    assert resumed["counts"]["completed"] == 4
    assert len(resumed["episodes"][1]["attempts"]) == 2
    assert len(vllm_starts) == 2


def test_batch_runs_episodes_with_requested_concurrency(tmp_path, monkeypatch) -> None:
    toolathlon_root = tmp_path / "toolathlon"
    tasks = [f"finalpool/task-{index}" for index in range(4)]
    for task in tasks:
        (toolathlon_root / "tasks" / task).mkdir(parents=True)
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
    manifest = batch.main(
        [
            "--tasks",
            *tasks,
            "--purpose",
            "trace-generation",
            "--concurrency",
            "4",
            "--bench-artifacts-dir",
            str(artifacts),
        ],
        repo_root=tmp_path,
        toolathlon_root=toolathlon_root,
        default_artifacts_dir=artifacts,
        default_image="image",
        default_model="decomposer-model",
        default_subagent_model="subagent-model",
        default_subagent_port=8030,
        start_vllm=lambda **kwargs: "process",
        stop_vllm=lambda process: None,
        docker=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )

    assert maximum_active == 4
    assert manifest["counts"]["completed"] == 4


def test_batch_distributes_episodes_across_external_vllm_ports(
    tmp_path, monkeypatch
) -> None:
    toolathlon_root = tmp_path / "toolathlon"
    tasks = [f"finalpool/task-{index}" for index in range(6)]
    for task in tasks:
        (toolathlon_root / "tasks" / task).mkdir(parents=True)
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
            "evaluation",
            "--subagent-ports",
            "18200",
            "18201",
            "18202",
            "--concurrency",
            "3",
            "--bench-artifacts-dir",
            str(artifacts),
        ],
        repo_root=tmp_path,
        toolathlon_root=toolathlon_root,
        default_artifacts_dir=artifacts,
        default_image="image",
        default_model="decomposer-model",
        default_subagent_model="subagent-model",
        default_subagent_port=8030,
        start_vllm=lambda **kwargs: starts.append(kwargs) or None,
        stop_vllm=lambda process: None,
        docker=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )

    assert [item["port"] for item in starts] == [18200, 18201, 18202]
    assert all(item["reuse"] for item in starts)
    assert Counter(ports) == Counter({18200: 2, 18201: 2, 18202: 2})
    assert manifest["config"]["subagent_ports"] == [18200, 18201, 18202]
    assert manifest["config"]["purpose"] == "evaluation"


def test_mlspace_serve_builds_one_replica_and_reverse_forward_per_gpu() -> None:
    args = SimpleNamespace(
        model="/models/qwen",
        max_model_len=65536,
        gpu_memory_utilization=0.9,
        ssh_key=Path("/secrets/key"),
        known_hosts=Path("/secrets/known_hosts"),
        hertz_port=44444,
        gpu_count=2,
        remote_port_start=18208,
        local_port_start=8030,
        hertz_user="matrosov",
        hertz_host="135.106.169.8",
    )

    vllm = mlspace_serve.vllm_command(args, 8031)
    tunnel = mlspace_serve.tunnel_command(args)

    assert vllm[vllm.index("--served-model-name") + 1] == mlspace_serve.SERVED_MODEL
    assert vllm[vllm.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking":false}'
    )
    forwards = [
        tunnel[index + 1]
        for index, value in enumerate(tunnel)
        if value == "-R"
    ]
    assert forwards == [
        "127.0.0.1:18208:127.0.0.1:8030",
        "127.0.0.1:18209:127.0.0.1:8031",
    ]


def test_mlspace_wait_fails_immediately_when_vllm_exits() -> None:
    process = SimpleNamespace(poll=lambda: 1, returncode=1)

    with pytest.raises(RuntimeError, match="vLLM exited with code 1"):
        mlspace_serve.wait_for_model(8030, mlspace_serve.SERVED_MODEL, 30, process)


def test_cleanup_continues_when_log_capture_fails(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_docker(*args, check=True):
        calls.append(args)
        if args[0] == "logs":
            raise OSError("capture failed")
        return subprocess.CompletedProcess(args, 0, "inspect", "")

    monkeypatch.setattr(run, "_docker", fake_docker)
    run._cleanup_episode(episode_dir=tmp_path, task_container="task-container")

    assert ("rm", "--force", "task-container") in calls
    cleanup = json.loads((tmp_path / "cleanup.json").read_text())
    assert cleanup["captures"][0]["error"] == "OSError('capture failed')"


def test_copy_user_configs_uses_container_copy_syntax(tmp_path, monkeypatch) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    source = configs / "global_configs.py"
    source.write_text("# local config\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(run, "TOOLATHLON_ROOT", tmp_path)
    monkeypatch.setattr(
        run,
        "USER_CONFIG_FILES",
        ("configs/global_configs.py",),
    )
    monkeypatch.setattr(
        run,
        "_docker",
        lambda *args, **kwargs: calls.append(args),
    )

    run._copy_user_configs("task-container")

    assert calls == [
        (
            "cp",
            str(source.resolve()),
            "task-container:/workspace/configs/global_configs.py",
        )
    ]


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
                "--tasks", "finalpool/alpha", "--purpose", "trace-generation",
                "--bench-artifacts-dir", str(artifacts),
            ],
            repo_root=tmp_path,
            toolathlon_root=toolathlon_root,
            default_artifacts_dir=artifacts,
            default_image="image",
            default_model="decomposer-model",
            default_subagent_model="subagent-model",
            default_subagent_port=8030,
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
    orphan = run_dir / "attempts" / "finalpool" / "alpha" / "rep-001" / "attempt-001"
    orphan.mkdir(parents=True)
    episode = {
        "key": "finalpool/alpha::rep-001",
        "task": "finalpool/alpha",
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


def test_execute_episode_maps_deterministic_trace_and_eval_paths(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "artifacts"
    run_dir = root / "runs" / "run-id"
    episode_id = batch.episode_id_for("run-id", "finalpool/alpha", 2, 1)
    artifact_dir = root / "traces" / "finalpool" / "alpha" / episode_id
    evaluation_path = root / "evals" / "finalpool" / "alpha" / episode_id / "result.json"
    artifact_dir.mkdir(parents=True)
    evaluation_path.parent.mkdir(parents=True)
    evaluation_path.write_text('{"pass": false}')

    class FakeProcess:
        def wait(self, timeout=None):
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
        subagent_port=8030,
        subagent_base_url=None,
        subagent_gpu="1",
        vllm_max_model_len=32768,
        subagent_recursion_limit=410,
        vllm_gpu_memory_utilization=0.8,
        vllm_startup_timeout=30,
        image="image",
        docker_socket=None,
        startup_timeout=10,
        n_jobs_per_worker=1000,
        max_steps=200,
        eval_config="scripts/formal_run_v0.json",
    )

    result = batch.execute_episode(
        args,
        runner_path=tmp_path / "run.py",
        root=root,
        run_dir=run_dir,
        episode={"task": "finalpool/alpha", "repetition": 2},
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
    assert command[command.index("--subagent-recursion-limit") + 1] == "410"
    assert command[command.index("--stashes-dir") + 1] == str(root / "stashes")
    assert command[command.index("--container-lock-file") + 1] == str(
        root / "runs" / "run-id" / "container.lock"
    )
