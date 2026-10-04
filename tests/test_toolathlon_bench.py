import asyncio
import importlib.util
import json
import os
import signal
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

import decomposer.agent_server
from gyms.toolathlon_bench import parallel, run as bench, task, usage
from gyms.toolathlon_bench import model_logging


BENCH_DIR = Path(bench.__file__).parent
CONTAINER_MODULES = ("agents", "tools", "model_logging", "task", "usage")


@pytest.fixture
def container_modules(monkeypatch):
    """Import bench modules as the container's Agent Server does: top-level."""
    monkeypatch.syspath_prepend(str(BENCH_DIR))
    for name in CONTAINER_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    for name in CONTAINER_MODULES:
        sys.modules.pop(name, None)


# --- run.py: one episode -------------------------------------------------------


IMAGE_MODEL = {
    "model_id": "container/model", "api_model": "image-model",
    "base_url": "http://image-provider/v1", "generation_config": {"temperature": 0.3},
}
BUNDLE = {
    "task_str": "Request",
    "task_dir": "finalpool/task",
    "resolved_task_config": {"task_dir": "finalpool/task"},
}


class FakeAgentServer:
    def __init__(self, command, *, stdout, stderr, episode_dir, events, exit_early):
        self.command = command
        self.returncode = 1 if exit_early else None
        self.events = events
        events.append(("agent_server", "start"))
        if not exit_early:
            (episode_dir / "runtime.json").write_text(json.dumps(
                {"agent_model": IMAGE_MODEL, "decomposer_model": IMAGE_MODEL}
            ))

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.events.append(("agent_server", "stopped"))
        self.returncode = 0
        return 0

    def kill(self):
        self.events.append(("agent_server", "killed"))
        self.returncode = -9


@pytest.fixture
def episode(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROXY_MASTER_KEY", "test-key")
    monkeypatch.delenv("LLM_PROXY_UNIX_SOCKET", raising=False)
    root = tmp_path / "toolathlon"
    (root / "tasks/finalpool/task").mkdir(parents=True)
    (root / "tasks/finalpool/k8s-mysql").mkdir(parents=True)
    configs = root / "configs"
    configs.mkdir()
    (configs / "global_configs_example.py").write_text("example = True\n")
    (configs / "token_key_session_example.py").write_text("example = True\n")
    (configs / "gcp-oauth.keys.json").write_text("{}")
    monkeypatch.setattr(bench, "TOOLATHLON_ROOT", root)
    monkeypatch.setattr(bench, "DOCKER_SOCKET", tmp_path / "docker.sock")
    (tmp_path / "docker.sock").touch()
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    monkeypatch.setattr(bench.tempfile, "mkdtemp", lambda prefix: str(private_dir))

    state = SimpleNamespace(
        tmp_path=tmp_path, root=root, private_dir=private_dir,
        episode_dir=tmp_path / "traces/task/episode",
        result_path=tmp_path / "evals/task/episode/result.json",
        events=[], docker=[], preprocess=CompletedProcess([], 0, "Preprocess done.", ""),
        eval_returncode=0, eval_writes_result=True, server_exits_early=False, agent_error=None,
        agent_state={
            "messages": [
                {"type": "human", "content": "Request"},
                {"type": "ai", "content": "", "tool_calls": [
                    {"id": "call", "name": "local-claim_done", "args": {}}
                ]},
                {"type": "ai", "content": "Answer"},
            ],
            "agent_runs": {"run": {"agent_id": "a", "status": "responded", "messages": [],
                                   "tool_calls": [{"id": "x", "name": "filesystem-read", "args": {"p": 1}}]}},
            "agents": {"a": {"agent_type_id": "qwen_3_5_4b_thinking"}},
            "decomposer_agent_runs": [{"status": "responded"}],
        },
    )

    def docker(*args, check=True):
        state.docker.append(args)
        state.events.append(("docker", args[0]))
        if args[:2] == ("image", "inspect"):
            return CompletedProcess(args, 0, "[]", "")
        if args[0] == "cp" and args[1].endswith(bench.CONTAINER_BUNDLE):
            Path(args[2]).write_text(json.dumps(BUNDLE))
        if "scripts.decoupled.container_preprocess" in args:
            state.events.append(("preprocess",))
            return state.preprocess
        if "scripts.decoupled.container_eval" in args:
            state.events.append(("eval",))
            trajectory = json.loads((state.episode_dir / "traj_log.json").read_text())
            trajectory["status"] = "success"
            (state.episode_dir / "traj_log.json").write_text(json.dumps(trajectory))
            if state.eval_writes_result:
                (state.episode_dir / "eval_res.json").write_text(json.dumps({"pass": state.eval_returncode == 0}))
            return CompletedProcess(args, state.eval_returncode, "Evaluation finished.", "")
        return CompletedProcess(args, 0, "", "")

    def guard(action, *args):
        state.events.append(("guard", action, args))
        return CompletedProcess(args, 0, f"{private_dir}/stash/artifact-1\n", "")

    def popen(command, *, stdout, stderr):
        state.server = FakeAgentServer(
            command, stdout=stdout, stderr=stderr, episode_dir=state.episode_dir,
            events=state.events, exit_early=state.server_exits_early,
        )
        return state.server

    async def capture(*args, **kwargs):
        state.events.append(("invoke", args, kwargs))
        return state.agent_state, state.agent_error

    response = MagicMock()
    response.__enter__.return_value.status = 200
    monkeypatch.setattr(bench, "_docker", docker)
    monkeypatch.setattr(bench, "_artifact_guard", guard)
    monkeypatch.setattr(bench.subprocess, "Popen", popen)
    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *args, **kwargs: response)
    monkeypatch.setattr(bench, "invoke_and_capture", capture)
    state.cleanup = MagicMock()
    monkeypatch.setattr(bench, "_cleanup_episode", state.cleanup)
    state.render = MagicMock(side_effect=lambda *args: state.events.append(("render",)))
    monkeypatch.setattr(bench, "write_trace_html", state.render)

    def run_episode(*argv):
        args = bench.create_parser().parse_args([
            *(argv or ["task"]), "--episode-id", "episode",
            "--artifacts-dir", str(tmp_path / "traces"), "--evals-dir", str(tmp_path / "evals"),
            "--startup-timeout", "5",
        ])
        return bench.run_episode(args)

    state.run_episode = run_episode
    return state


def test_episode_runs_toolathlon_phases_in_trusted_order(episode):
    episode.run_episode()

    order = [event[0] if event[0] != "guard" else f"guard:{event[1]}" for event in episode.events
             if event[0] in {"preprocess", "guard", "agent_server", "invoke", "eval", "render"}]
    assert order == ["preprocess", "guard:stash", "agent_server", "invoke", "agent_server",
                     "guard:restore", "eval", "render"]
    run_args = next(args for args in episode.docker if args[0] == "run")
    assert run_args[run_args.index("--network") + 1] == "host"
    assert run_args[run_args.index("--env") + 1] == "LLM_PROXY_MASTER_KEY"
    assert f"{episode.episode_dir}:/workspace/dumps" in run_args
    assert f"{episode.tmp_path / 'docker.sock'}:/var/run/docker.sock" in run_args
    assert "test-key" not in " ".join(run_args)
    copied = [args for args in episode.docker if args[0] == "cp" and args[1].startswith(str(episode.root))]
    assert [args[2].split(":", 1)[1] for args in copied] == ["/workspace/configs/global_configs.py",
                                                              "/workspace/configs/gcp-oauth.keys.json",
                                                              "/workspace/configs/token_key_session.py"]
    preprocess = next(args for args in episode.docker if "scripts.decoupled.container_preprocess" in args)
    assert preprocess[preprocess.index("--task_dir") + 1] == "finalpool/task"
    assert preprocess[preprocess.index("--host_output_folder") + 1] == str(episode.episode_dir)

    stash = next(event for event in episode.events if event[:2] == ("guard", "stash"))[2]
    assert stash[stash.index("--task-path") + 1] == "/workspace/tasks/finalpool/task"
    restore = next(event for event in episode.events if event[:2] == ("guard", "restore"))[2]
    assert restore[restore.index("--stash-dir") + 1] == f"{episode.private_dir}/stash/artifact-1"
    command = episode.server.command
    assert command[-2:] == ["/opt/agents/bin/python", "/opt/decomposer/gyms/toolathlon_bench/task.py"]
    assert f"TOOLATHLON_BUNDLE={bench.CONTAINER_BUNDLE}" in command

    invoke = next(event for event in episode.events if event[0] == "invoke")
    url, assistant, inputs = invoke[1]
    assert url.startswith("http://127.0.0.1:") and assistant == "decomposer"
    assert inputs == {"messages": [{"role": "user", "content": "Request"}]}
    assert invoke[2]["config"] == {"recursion_limit": 410}

    evaluation = json.loads(episode.result_path.read_text())
    assert evaluation["pass"] is True and evaluation["native_pass"] is True
    assert evaluation["native_result"] == {"pass": True}
    trace = json.loads((episode.episode_dir / "trace.json").read_text())
    assert trace["harness"] == "decomposer"
    assert trace["model"] == trace["agent_model_id"] == "container/model"
    assert trace["agent_api_model"] == "image-model" and trace["teacher_backend"] == "container"
    assert trace["agent_runs"] == episode.agent_state["agent_runs"]
    usage_summary = json.loads((episode.episode_dir / "usage.json").read_text())
    assert usage_summary["totals"]["agent_runs"] == 1
    assert (episode.episode_dir / "answer.txt").read_text() == "Answer"
    assert json.loads((episode.episode_dir / "traj_log.json").read_text())["status"] == "success"
    assert (episode.root / "configs/global_configs.py").read_text() == "example = True\n"
    episode.cleanup.assert_called_once_with(
        episode_dir=episode.episode_dir, task_container=run_args[run_args.index("--name") + 1], task="task",
    )
    assert not episode.private_dir.exists()


def test_react_episode_reports_react_usage(episode):
    episode.agent_state.pop("agent_runs")
    episode.run_episode("task", "--agent", "qwen_3_5_4b_thinking")
    trace = json.loads((episode.episode_dir / "trace.json").read_text())
    assert trace["harness"] == "react" and trace["decomposer_model"] is None
    assert trace["teacher_backend"] == "agent"
    assert "react" in json.loads((episode.episode_dir / "usage.json").read_text())


def test_failed_agent_keeps_native_score_and_trajectory_status(episode):
    failure = RuntimeError("model failed")
    episode.agent_error = failure
    with pytest.raises(RuntimeError, match="model failed") as caught:
        episode.run_episode()
    assert caught.value.__cause__ is failure
    evaluation = json.loads(episode.result_path.read_text())
    assert evaluation["pass"] is False and evaluation["native_pass"] is True
    trajectory = json.loads((episode.episode_dir / "traj_log.json").read_text())
    assert trajectory["status"] == "failed" and "model failed" in trajectory["error"]
    episode.cleanup.assert_called_once()


def test_unconfirmed_agent_shutdown_skips_restore_and_evaluation(episode):
    episode.agent_state["agent_shutdown_error"] = "TimeoutError()"
    episode.run_episode()
    assert not any(event[:2] == ("guard", "restore") for event in episode.events)
    assert ("eval",) not in episode.events
    evaluation = json.loads(episode.result_path.read_text())
    assert evaluation["pass"] is False and evaluation["native_pass"] is None


@pytest.mark.parametrize(("returncode", "output", "message"), [
    (1, "boom", "exited with code 1"),
    (0, "Connection refused by canvas", "connection refused"),
])
def test_preprocess_failure_stops_before_hiding_artifacts(episode, returncode, output, message):
    episode.preprocess = CompletedProcess([], returncode, output, "")
    with pytest.raises(RuntimeError, match=message):
        episode.run_episode()
    assert not any(event[0] in {"guard", "agent_server"} for event in episode.events)
    assert (episode.episode_dir / "preprocess.log").read_text() == output
    episode.cleanup.assert_called_once()
    assert not episode.private_dir.exists()


def test_agent_server_exit_during_startup_fails_episode(episode):
    episode.server_exits_early = True
    with pytest.raises(RuntimeError, match="Agent Server exited"):
        episode.run_episode()
    assert not any(event[0] == "invoke" for event in episode.events)
    episode.cleanup.assert_called_once()


def test_evaluator_failure_without_result_file(episode):
    episode.eval_returncode = 1
    episode.eval_writes_result = False
    episode.run_episode()
    evaluation = json.loads(episode.result_path.read_text())
    assert evaluation["pass"] is False and evaluation["native_pass"] is False
    assert evaluation["native_result"] is None and evaluation["returncode"] == 1


def test_container_lock_is_released_after_failed_setup(episode, tmp_path):
    episode.preprocess = CompletedProcess([], 1, "boom", "")
    lock = tmp_path / "locks/container.lock"
    with pytest.raises(RuntimeError):
        episode.run_episode("task", "--container-lock-file", str(lock))
    assert lock.is_file()
    episode.cleanup.assert_called_once()


def test_agent_server_startup_timeout_kills_server(episode, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(bench.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(bench.time, "sleep", lambda seconds: None)
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(bench.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError, match="did not become ready"):
        episode.run_episode()
    assert ("agent_server", "killed") in episode.events
    assert not any(event[0] == "invoke" for event in episode.events)


def test_parser_rejects_invalid_limits(episode):
    for flag in ("--n-jobs-per-worker", "--agent-timeout"):
        with pytest.raises(SystemExit):
            episode.run_episode("task", flag, "0")


def test_missing_proxy_socket_fails_before_docker(episode, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROXY_UNIX_SOCKET", str(tmp_path / "missing.sock"))
    with pytest.raises(RuntimeError, match="Model proxy socket is missing"):
        episode.run_episode()
    assert episode.docker == []


def test_episode_requires_key_and_known_task(episode, monkeypatch):
    with pytest.raises(ValueError, match="Unknown Toolathlon task"):
        episode.run_episode("../task")
    monkeypatch.delenv("LLM_PROXY_MASTER_KEY")
    with pytest.raises(RuntimeError, match="LLM_PROXY_MASTER_KEY"):
        episode.run_episode()
    assert episode.docker == []


def test_proxy_socket_is_mounted_read_only(episode, monkeypatch, tmp_path):
    import socket

    path = tmp_path / "proxy.sock"
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(path))
    monkeypatch.setenv("LLM_PROXY_UNIX_SOCKET", str(path))
    try:
        episode.run_episode()
    finally:
        listener.close()
    run_args = next(args for args in episode.docker if args[0] == "run")
    assert f"{tmp_path}:/run/model-proxy:ro" in run_args
    assert "LLM_PROXY_UNIX_SOCKET=/run/model-proxy/proxy.sock" in run_args


def test_prepare_configs_keeps_user_files(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "global_configs_example.py").write_text("example\n")
    (configs / "token_key_session_example.py").write_text("example\n")
    (configs / "global_configs.py").write_text("real\n")
    monkeypatch.setattr(bench, "TOOLATHLON_ROOT", tmp_path)
    bench._prepare_configs()
    assert (configs / "global_configs.py").read_text() == "real\n"
    assert (configs / "token_key_session.py").read_text() == "example\n"
    assert (configs / ".mcp-auth").is_dir()


def test_trajectory_uses_toolathlon_log_format():
    state = {
        "messages": [
            HumanMessage("Request"),
            AIMessage("", tool_calls=[{"id": "c", "name": "local-claim_done", "args": {}}]),
            ToolMessage("done", tool_call_id="c"),
            {"type": "ai", "content": [{"type": "text", "text": "Answer"}]},
        ],
        "agent_runs": {"r": {"tool_calls": [{"id": "x", "name": "pdf-read", "args": {"page": 1}}]}},
    }
    log = bench.trajectory(state, bundle=BUNDLE, episode_id="e", started_at="start", error=None)
    assert log["config"] == BUNDLE["resolved_task_config"]
    assert log["task_dir"] == "finalpool/task" and log["session_id"] == "e"
    assert [message["role"] for message in log["messages"]] == [
        "user", "assistant", "tool", "assistant", "agent_tool_call",
    ]
    assert log["messages"][1]["tool_calls"] == [{"id": "c", "name": "local-claim_done", "args": {}}]
    assert json.loads(log["messages"][3]["content"]) == [{"type": "text", "text": "Answer"}]
    assert log["messages"][4] == {"role": "agent_tool_call", "agent_run_id": "r",
                                  "content": '{"name": "pdf-read", "args": {"page": 1}}'}
    assert log["tool_calls"]["tools"] == ["local-claim_done", "pdf-read"]
    assert log["key_stats"] == {"interaction_turns": 2, "tool_calls": 1,
                                "agent_llm_requests": 2, "agent_runs": 1}
    assert log["status"] == "success" and "error" not in log
    failed = bench.trajectory({}, bundle=BUNDLE, episode_id="e", started_at="start", error="boom")
    assert failed["status"] == "failed" and failed["error"] == "boom"


def test_cleanup_stops_kubernetes_task_and_redacts_inspection(tmp_path, monkeypatch):
    calls = []

    def docker(*args, check=True):
        calls.append(args)
        if args[0] == "inspect":
            return CompletedProcess(args, 0, json.dumps([{
                "Id": "id", "Name": "name", "Image": "image",
                "Config": {"Env": ["LLM_PROXY_MASTER_KEY=secret"]},
                "State": {"Status": "running", "ExitCode": 0, "Pid": 7},
            }]), "")
        return CompletedProcess(args, 0, "log line\n", "")

    monkeypatch.setattr(bench, "_docker", docker)
    bench._cleanup_episode(episode_dir=tmp_path, task_container="c", task="k8s-mysql")
    assert calls[0] == ("exec", "--workdir", "/workspace", "c", *bench.K8S_TASK_CLEANUP_COMMANDS["k8s-mysql"])
    assert calls[-1] == ("rm", "--force", "--volumes", "c")
    inspection = (tmp_path / "task.inspect.json").read_text()
    assert "secret" not in inspection and json.loads(inspection)[0]["State"]["Status"] == "running"
    assert (tmp_path / "task.log").read_text() == "log line\n"
    cleanup = json.loads((tmp_path / "cleanup.json").read_text())
    assert cleanup["task_cleanup"]["returncode"] == 0
    calls.clear()
    bench._cleanup_episode(episode_dir=tmp_path, task_container="c", task="task")
    assert calls[0][0] == "logs"


def test_stop_agent_server_signals_only_a_running_server(monkeypatch):
    calls = []
    monkeypatch.setattr(bench, "_exec", lambda *args, check=True: calls.append(args))
    server = MagicMock()
    server.poll.return_value = None
    bench._stop_agent_server("c", server)
    assert calls == [("c", "sh", "-c", f'kill -TERM "$(cat {task.PID_FILE})"')]
    server.wait.assert_called_once_with(timeout=60)
    calls.clear()
    server.poll.return_value = 0
    bench._stop_agent_server("c", server)
    assert calls == []


def test_artifact_guard_runs_toolathlon_module(monkeypatch):
    run = MagicMock(return_value=CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(bench.subprocess, "run", run)
    bench._artifact_guard("stash", "--container", "c")
    command = run.call_args.args[0]
    assert command[1:] == ["-m", "scripts.containerized.task_artifact_guard", "stash", "--container", "c"]
    assert run.call_args.kwargs["env"]["PYTHONPATH"] == str(bench.TOOLATHLON_ROOT)
    assert run.call_args.kwargs["check"] is True


def test_docker_reports_failures(monkeypatch):
    monkeypatch.setattr(bench.subprocess, "run",
                        lambda *args, **kwargs: CompletedProcess(args, 3, "", "no such image"))
    with pytest.raises(RuntimeError, match="no such image"):
        bench._docker("image", "inspect", "missing")
    assert bench._docker("image", "inspect", "missing", check=False).returncode == 3


def test_main_dispatches_episode_or_parallel(monkeypatch):
    episode_run = MagicMock()
    parallel_run = MagicMock()
    monkeypatch.setattr(bench, "run_episode", episode_run)
    monkeypatch.setattr(parallel, "run", parallel_run)
    bench.main(["task", "--episode-id", "e"])
    episode_run.assert_called_once()
    bench.main(["--all"])
    parallel_run.assert_called_once()


# --- parallel.py ---------------------------------------------------------------


def test_select_tasks_from_task_pool(tmp_path):
    for name in ("b", "a", ".hidden"):
        (tmp_path / name).mkdir()
    (tmp_path / "file.txt").touch()
    assert parallel.select_tasks(tmp_path, all_tasks=True) == ["a", "b"]
    assert parallel.select_tasks(tmp_path, task="a") == ["a"]
    with pytest.raises(ValueError):
        parallel.select_tasks(tmp_path, tasks=["a", "missing"])
    with pytest.raises(ValueError):
        parallel.select_tasks(tmp_path, task="a", all_tasks=True)


def test_parallel_runs_each_repetition_as_an_episode(tmp_path, monkeypatch):
    (tmp_path / "toolathlon/tasks/finalpool/task").mkdir(parents=True)
    monkeypatch.setattr(bench, "TOOLATHLON_ROOT", tmp_path / "toolathlon")
    commands = []

    def execute(command, directory, timeout, stop_event):
        commands.append(command)
        return 0, False

    monkeypatch.setattr(parallel, "execute", execute)
    args = bench.create_parser().parse_args(["task", "-n", "2", "--output-dir", str(tmp_path / "raw"),
                                             "--agent", "qwen_3_5_4b_thinking"])
    root = parallel.run(args)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["harness"] == "react"
    assert sorted(episode["repetition"] for episode in manifest["episodes"]) == [1, 2]
    command = commands[0]
    assert command[1] == bench.__file__ and command[2] == "task"
    assert command[command.index("--agent") + 1] == "qwen_3_5_4b_thinking"
    assert command[command.index("--artifacts-dir") + 1] == str(root / "traces")
    assert "--tasks" not in command and "--repetitions" not in command


def test_execute_preserves_logs_and_times_out(tmp_path):
    import threading

    code, timed_out = parallel.execute([sys.executable, "-c", "print('hi')"], tmp_path, 30, threading.Event())
    assert (code, timed_out) == (0, False)
    assert (tmp_path / "runner.stdout.log").read_text() == "hi\n"
    code, timed_out = parallel.execute([sys.executable, "-c", "import time; time.sleep(30)"],
                                       tmp_path, 0.1, threading.Event())
    assert timed_out


# --- task.py: container-side Agent Server --------------------------------------


@pytest.fixture
def serve_env(tmp_path, monkeypatch, container_modules):
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps({
        "system_prompts": {"agent": "Workspace: /workspace/dumps/workspace"},
        "container_paths": {"agent_workspace": "/workspace/dumps/workspace"},
    }))
    monkeypatch.setenv("TOOLATHLON_BUNDLE", str(bundle_path))
    monkeypatch.setenv("TOOLATHLON_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GATEWAY_PORT", "4001")
    monkeypatch.setenv("AGENT_SERVER_PORT", "4002")
    monkeypatch.setenv("N_JOBS_PER_WORKER", "7")
    monkeypatch.setattr(task, "PID_FILE", tmp_path / "server.pid")
    monkeypatch.setattr(task, "model_metadata", lambda model_id: {"model_id": model_id})
    gateway = SimpleNamespace(pid=123, returncode=None)

    async def wait():
        gateway.returncode = 0

    gateway.wait = wait
    spawned = []

    async def create_subprocess_exec(*command, **kwargs):
        spawned.append((command, kwargs))
        return gateway

    killed = []
    monkeypatch.setattr(task.asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr(task.os, "killpg", lambda pid, signum: killed.append((pid, signum)))
    return SimpleNamespace(tmp_path=tmp_path, bundle_path=bundle_path, spawned=spawned,
                           killed=killed, gateway=gateway)


def test_serve_hides_bundle_then_runs_agent_server_until_signal(serve_env, monkeypatch):
    monkeypatch.setattr(task, "wait_ready", AsyncMock())
    seen = {}

    @asynccontextmanager
    async def agent_server(config, **kwargs):
        seen.update(kwargs, config=config, pid=task.PID_FILE.read_text(),
                    bundle_exists=serve_env.bundle_path.exists(),
                    runtime=json.loads((serve_env.tmp_path / "runtime.json").read_text()))
        os.kill(os.getpid(), signal.SIGTERM)
        yield "http://127.0.0.1:4002"

    monkeypatch.setattr(decomposer.agent_server, "agent_server", agent_server)
    asyncio.run(task.serve())

    command, kwargs = serve_env.spawned[0]
    assert command[:3] == ("uv", "run", "python") and command[3].endswith("tool_gateway.py")
    assert command[4:] == ("--bundle-file", str(serve_env.bundle_path), "--port", "4001")
    assert kwargs["cwd"] == "/workspace" and kwargs["start_new_session"] is True
    task.wait_ready.assert_awaited_once()
    assert task.wait_ready.await_args.args[0] == "http://127.0.0.1:4001/health"
    assert seen["config"] == BENCH_DIR / "langgraph.json"
    assert seen["port"] == 4002 and seen["n_jobs_per_worker"] == 7
    assert seen["pid"] == str(os.getpid()) and seen["bundle_exists"] is False
    assert seen["runtime"] == {
        "agent_system_prompt": "Workspace: /workspace/dumps/workspace",
        "agent_workspace": "/workspace/dumps/workspace",
        "gateway_url": "http://127.0.0.1:4001/sse",
        "agent_server_url": "http://127.0.0.1:4002",
        "agent_model": {"model_id": "lmrouter/qwen_3_5_4b_unlooped_thinking"},
        "decomposer_model": {"model_id": "lmrouter/qwen_3_8_flash_next_non_thinking"},
    }
    assert serve_env.killed == [(123, signal.SIGTERM)]
    assert not task.PID_FILE.exists()


def test_serve_stops_gateway_when_it_never_becomes_ready(serve_env, monkeypatch):
    monkeypatch.setattr(task, "wait_ready", AsyncMock(side_effect=TimeoutError("not ready")))
    with pytest.raises(TimeoutError):
        asyncio.run(task.serve())
    assert serve_env.bundle_path.exists()
    assert not (serve_env.tmp_path / "runtime.json").exists()
    assert serve_env.killed == [(123, signal.SIGTERM)]
    assert not task.PID_FILE.exists()


def test_serve_kills_gateway_that_ignores_sigterm(serve_env, monkeypatch):
    monkeypatch.setattr(task, "wait_ready", AsyncMock(side_effect=TimeoutError("not ready")))
    wait_for = task.asyncio.wait_for

    async def expire(awaitable, timeout):
        if timeout == 30:
            awaitable.close()
            raise TimeoutError
        return await wait_for(awaitable, timeout)

    monkeypatch.setattr(task.asyncio, "wait_for", expire)
    with pytest.raises(TimeoutError):
        asyncio.run(task.serve())
    assert serve_env.killed == [(123, signal.SIGTERM), (123, signal.SIGKILL)]


def _mock_async_client(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(task.httpx, "AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def test_wait_ready_polls_until_healthy(monkeypatch):
    responses = iter([None, 503, 200])

    def handler(request):
        status = next(responses)
        if status is None:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(status)

    _mock_async_client(monkeypatch, handler)
    asyncio.run(task.wait_ready("http://gateway/health", SimpleNamespace(returncode=None), timeout=5))


def test_wait_ready_fails_fast_or_times_out(monkeypatch):
    _mock_async_client(monkeypatch, lambda request: httpx.Response(503))
    with pytest.raises(RuntimeError, match="exited"):
        asyncio.run(task.wait_ready("http://gateway/health", SimpleNamespace(returncode=1), timeout=5))
    with pytest.raises(TimeoutError):
        asyncio.run(task.wait_ready("http://gateway/health", SimpleNamespace(returncode=None), timeout=0.3))


def test_model_metadata_has_sampling_without_credentials(monkeypatch):
    monkeypatch.setenv("LLM_PROXY_MASTER_KEY", "test-secret")
    metadata = task.model_metadata("lmrouter/qwen_3_5_4b_unlooped_thinking")
    assert metadata["api_model"] == "Qwen/Qwen3.5-4B-unlooped"
    assert metadata["base_url"] == "https://lmrouter.2a2i.org/v1"
    assert metadata["generation_config"]["preserve_reasoning"] is True
    assert "test-secret" not in json.dumps(metadata)


# --- tools.py and agents.py: Agent Server graphs -------------------------------


def test_tools_start_from_runtime_and_gateway(tmp_path, monkeypatch, container_modules):
    import tools

    runtime = {"gateway_url": "http://127.0.0.1:4001/sse", "agent_workspace": str(tmp_path)}
    (tmp_path / "runtime.json").write_text(json.dumps(runtime))
    monkeypatch.setenv("TOOLATHLON_DATA_DIR", str(tmp_path))
    opened = {}

    @asynccontextmanager
    async def sse_client(url, *, sse_read_timeout):
        opened["url"] = url
        yield "read", "write"

    class Session:
        def __init__(self, read, write, *, read_timeout_seconds):
            opened["streams"] = (read, write)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        initialize = AsyncMock()

    monkeypatch.setattr(tools, "sse_client", sse_client)
    monkeypatch.setattr(tools, "ClientSession", Session)
    monkeypatch.setattr(tools, "load_mcp_tools", AsyncMock(return_value=["gateway-tool"]))

    async def check():
        with pytest.raises(RuntimeError):
            tools.get_tools()
        async with tools.lifespan(tools.app):
            assert tools.get_tools() == ["gateway-tool"]
            assert tools.get_runtime() == runtime
        with pytest.raises(RuntimeError):
            tools.get_runtime()

    asyncio.run(check())
    assert opened == {"url": runtime["gateway_url"], "streams": ("read", "write")}
    assert tools.load_mcp_tools.await_args.kwargs == {"server_name": "gateway"}


def test_tool_output_truncation_saves_full_output(tmp_path, monkeypatch, container_modules):
    import tools

    monkeypatch.setattr(tools.app.state, "runtime", {"agent_workspace": str(tmp_path)}, raising=False)
    request = SimpleNamespace(tool_call={"id": "call"})
    middleware = tools.truncate_mcp_tool_output

    async def call(handler):
        return await middleware.awrap_tool_call(request, handler)

    short = ToolMessage("ok", tool_call_id="call")
    assert asyncio.run(call(AsyncMock(return_value=short))) is short

    long = ToolMessage("x" * (tools.MAX_TOOL_OUTPUT_CHARS + 5), tool_call_id="call")
    truncated = asyncio.run(call(AsyncMock(return_value=long)))
    assert truncated.content.startswith("x" * tools.MAX_TOOL_OUTPUT_CHARS + "\n...[truncated")
    saved = list((tmp_path / ".overlong_tool_outputs").iterdir())
    assert len(saved) == 1 and saved[0].read_text() == long.content
    assert saved[0].stem in truncated.content

    failed = asyncio.run(call(AsyncMock(side_effect=ValueError("bad args"))))
    assert failed.status == "error" and failed.content == "Tool call failed: bad args"


def test_agents_use_task_prompt_tools_and_server_url(monkeypatch, container_modules):
    import agents
    import tools

    @tool
    def gateway_tool() -> str:
        """Gateway tool."""
        return ""

    monkeypatch.setattr(tools.app.state, "tools", [gateway_tool], raising=False)
    monkeypatch.setattr(tools.app.state, "runtime", {
        "agent_system_prompt": "Task prompt", "agent_server_url": "http://127.0.0.1:4002",
    }, raising=False)
    create_agent = MagicMock(return_value="agent")
    create_decomposer = MagicMock(return_value="decomposer")
    monkeypatch.setattr(agents, "create_agent", create_agent)
    monkeypatch.setattr(agents, "create_decomposer_agent", create_decomposer)

    assert agents.qwen_3_5_4b_thinking() == "agent"
    kwargs = create_agent.call_args.kwargs
    assert kwargs["model"].model_name == "Qwen/Qwen3.5-4B-unlooped"
    assert kwargs["tools"] == [gateway_tool] and kwargs["system_prompt"] == "Task prompt"
    assert len(kwargs["middleware"]) == 2

    assert agents.decomposer() == "decomposer"
    kwargs = create_decomposer.call_args.kwargs
    assert kwargs["decomposer_model"].model_name == "Qwen/Qwen3.8-Flash-Next-NVFP4"
    assert kwargs["agent_types"][0]["url"] == "http://127.0.0.1:4002"
    assert kwargs["agent_types"][0]["assistant_id"] == "qwen_3_5_4b_thinking"
    assert kwargs["agent_recursion_limit"] == 410
    graphs = json.loads((BENCH_DIR / "langgraph.json").read_text())["graphs"]
    assert set(graphs) == {"qwen_3_5_4b_thinking", "decomposer"}


# --- tool_gateway.py: Toolathlon MCP gateway with native local tools ------------


@pytest.fixture
def gateway_module(monkeypatch):
    class ToolRecord:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    class Registry:
        def __init__(self):
            self._records = {}

        def _allocate_name(self, name, prefix, always_prefix):
            assert prefix == "local" and always_prefix is False
            return name

    class UpstreamGateway:
        def __init__(self, bundle_file, debug=False):
            self.bundle_file, self.debug = bundle_file, debug
            self.bundle = {}
            self.registry = Registry()

        async def startup(self, app):
            self.bundle = json.loads(Path(self.bundle_file).read_text())

        async def _remote_call(self, record, arguments):
            return {"remote": record.exposed_name, "arguments": arguments}

        def create_app(self):
            return "app"

    def local_tool(name, result):
        async def invoke(context, params):
            return {"context": context.context, "params": json.loads(params), "result": result}
        return SimpleNamespace(name=name, description=f"{name} tool",
                               params_json_schema={"type": "object"}, on_invoke_tool=invoke)

    upstream = types.ModuleType("scripts.decoupled.container_tool_gateway")
    upstream.ContainerToolGateway = UpstreamGateway
    upstream.ToolRecord = ToolRecord
    task_agent = types.ModuleType("utils.roles.task_agent")
    task_agent.local_tool_mappings = {
        "python_execute": local_tool("local-python-execute", "ran"),
        "handle_overlong_tool_outputs": [local_tool("local-search_overlong_tooloutput", "found")],
        "claim_done": local_tool("local-claim_done", "done"),
        "web_search": local_tool("local-web_search", "searched"),
    }
    openai_agents = types.ModuleType("agents")
    openai_agents.RunContextWrapper = lambda context: SimpleNamespace(context=context)
    modules = {
        "scripts": types.ModuleType("scripts"),
        "scripts.decoupled": types.ModuleType("scripts.decoupled"),
        "scripts.decoupled.container_tool_gateway": upstream,
        "utils": types.ModuleType("utils"),
        "utils.roles": types.ModuleType("utils.roles"),
        "utils.roles.task_agent": task_agent,
        "agents": openai_agents,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", ["placeholder", *sys.path])
    spec = importlib.util.spec_from_file_location("tool_gateway", BENCH_DIR / "tool_gateway.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert sys.path[0] == "/workspace"
    return module


def test_gateway_exposes_requested_native_local_tools(gateway_module, tmp_path):
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "needed_local_tools": ["python_execute", "handle_overlong_tool_outputs", "claim_done"],
        "container_paths": {"agent_workspace": "/workspace/dumps/workspace"},
    }))
    gateway = gateway_module.ContainerToolGateway(str(bundle))
    assert gateway.debug is True
    asyncio.run(gateway.startup(None))
    assert sorted(gateway.registry._records) == [
        "local-python-execute", "local-search_overlong_tooloutput",
    ]
    record = gateway.registry._records["local-python-execute"]
    assert record.backend_type == "native_local" and record.input_schema == {"type": "object"}

    result = asyncio.run(gateway._remote_call(record, {"code": "print(1)"}))
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"context": {"_agent_workspace": "/workspace/dumps/workspace"},
                       "params": {"code": "print(1)"}, "result": "ran"}

    remote = SimpleNamespace(backend_type="mcp", exposed_name="filesystem-read")
    assert asyncio.run(gateway._remote_call(remote, {"p": 1})) == {
        "remote": "filesystem-read", "arguments": {"p": 1},
    }


def test_gateway_main_serves_on_loopback(gateway_module, monkeypatch):
    run_app = MagicMock()
    monkeypatch.setattr(gateway_module.web, "run_app", run_app)
    monkeypatch.setattr(sys, "argv", ["tool_gateway.py", "--bundle-file", "/run/b.json", "--port", "4001"])
    gateway_module.main()
    run_app.assert_called_once_with("app", host="127.0.0.1", port=4001)


# --- usage.py and model_logging.py ---------------------------------------------


def test_usage_summary_totals_decomposer_and_agents():
    def ai(tokens, cost=None):
        return {"type": "ai", "usage_metadata": {"input_tokens": tokens, "output_tokens": 1, "total_tokens": tokens + 1},
                "response_metadata": {"token_usage": {"cost": cost}}}

    summary = usage.build_usage_summary(
        [ai(10, 0.5), {"type": "human"}],
        {"r": {"agent_id": "a", "status": "responded", "messages": [ai(3), {"type": "ai"}]}},
        {"a": {"agent_type_id": "qwen"}},
    )
    assert summary["decomposer"]["input_tokens"] == 10
    assert summary["agents"]["r"]["agent_type_id"] == "qwen"
    assert summary["agents"]["r"]["model_responses"] == 2
    assert summary["agents"]["r"]["responses_with_usage"] == 1
    assert summary["totals"]["input_tokens"] == 13 and summary["totals"]["cost"] == 0.5
    assert summary["totals"]["agent_runs"] == 1
    assert usage.build_usage_summary([], {})["totals"]["cost"] is None


def test_durable_model_call_log_records_success_and_error(tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv(model_logging.LOG_PATH_ENV, str(log))
    request = SimpleNamespace(
        model=SimpleNamespace(model_name="m"), system_message=None,
        messages=[HumanMessage("Request")],
    )
    middleware = model_logging.durable_model_call_log
    response = SimpleNamespace(result=[AIMessage("Answer")])
    assert asyncio.run(middleware.awrap_model_call(request, AsyncMock(return_value=response))) is response
    with pytest.raises(ValueError):
        asyncio.run(middleware.awrap_model_call(request, AsyncMock(side_effect=ValueError("down"))))
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [record["status"] for record in records] == ["success", "error"]
    assert records[0]["model"] == "m" and records[0]["request_message_count"] == 1
    assert records[0]["response"][0]["data"]["content"] == "Answer"
    assert records[1]["error"] == "ValueError('down')"
