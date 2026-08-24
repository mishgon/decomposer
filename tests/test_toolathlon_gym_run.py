import json
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from gyms.toolathlon_gym import batch, mlspace_serve, run


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
        ((["docker", "ps"],), {"check": False, "capture_output": True, "text": True})
    ]


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


def test_postgres_image_is_fully_qualified_for_podman() -> None:
    assert run.POSTGRES_IMAGE == "docker.io/library/postgres:15"


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


def test_main_requires_explicit_trace_generation_purpose(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run.py", "example"])

    with pytest.raises(SystemExit):
        run.main()


def test_main_rejects_path_traversal(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "../finalpool", "--purpose", "trace-generation"],
    )

    with pytest.raises(ValueError, match="Unknown Toolathlon task"):
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


def test_batch_runs_episodes_with_requested_concurrency(tmp_path, monkeypatch) -> None:
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

    assert maximum_active == 4
    assert manifest["counts"]["completed"] == 4


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


def test_mlspace_serve_builds_one_replica_and_reverse_forward_per_gpu() -> None:
    args = SimpleNamespace(
        model="/models/gemma",
        max_model_len=65536,
        gpu_memory_utilization=0.9,
        ssh_key=Path("/secrets/key"),
        known_hosts=Path("/secrets/known_hosts"),
        hertz_port=44444,
        gpu_count=2,
        remote_port_start=18108,
        local_port_start=8023,
        hertz_user="matrosov",
        hertz_host="135.106.169.8",
    )

    vllm = mlspace_serve.vllm_command(args, 8024)
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
        "127.0.0.1:18108:127.0.0.1:8023",
        "127.0.0.1:18109:127.0.0.1:8024",
    ]


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

    assert ("rm", "--force", "task-container") in calls
    assert ("rm", "--force", "pg-container") in calls
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
