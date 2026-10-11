import asyncio
import json
from contextlib import asynccontextmanager
from subprocess import CompletedProcess
from unittest.mock import AsyncMock, MagicMock

import pytest

import decomposer.visualization as visualization
from examples.minimal import run as minimal
from gyms.toolathlon_gym import run as gym
from gyms.toolathlon_gym.task import model_metadata


@pytest.mark.parametrize("agent_failed,skipped", [(False, False), (True, False), (False, True)])
def test_render_failure_preserves_evaluation_and_container_metadata(tmp_path, monkeypatch, caplog, agent_failed, skipped):
    monkeypatch.delenv("LLM_PROXY_MASTER_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-secret")
    monkeypatch.setenv("https_proxy", "http://user:test-proxy-secret@proxy:8080")
    monkeypatch.setenv("no_proxy", "lower.example")
    monkeypatch.setenv("NO_PROXY", "upper.example")
    monkeypatch.delenv("LLM_PROXY_UNIX_SOCKET", raising=False)
    monkeypatch.setattr(gym, "TOOLATHLON_ROOT", tmp_path / "toolathlon")
    (gym.TOOLATHLON_ROOT / "tasks/finalpool/task").mkdir(parents=True)
    episode = tmp_path / "traces/task/episode"
    result_path = tmp_path / "evals/task/episode/result.json"
    image_model = {
        "model_id": "container/model", "api_model": "image-model",
        "base_url": "http://image-provider/v1", "generation_config": {"temperature": 0.3},
    }
    runtime = {
        "agent_model": image_model, "decomposer_model": image_model,
        "task_config": {
            "task_str": "Request", "agent_workspace": "/workspace/agent", "launch_time": None,
            "evaluation": {"evaluation_command": "python evaluate.py", "groundtruth_workspace": None},
        },
    }

    def docker(*args, check=True):
        output = ""
        if "{{.State.Health.Status}}" in args:
            output = "healthy"
        elif "/proc/1/comm" in args:
            output = "postgres"
        elif "SELECT to_regclass('email.messages') IS NOT NULL;" in args:
            output = "t"
        elif args[0] == "port":
            output = "127.0.0.1:2024"
        elif args[0] == "run" and args[-1] == gym.DEFAULT_IMAGE:
            assert ("--env", "OPENROUTER_API_KEY") in list(zip(args, args[1:]))
            assert "test-openrouter-secret" not in " ".join(args)
            assert ("--env", "https_proxy") in list(zip(args, args[1:]))
            assert "test-proxy-secret" not in " ".join(args)
            for name in ("no_proxy", "NO_PROXY"):
                value = next(arg.split("=", 1)[1] for arg in args if arg.startswith(name + "="))
                assert set(value.split(",")) >= {
                    "lower.example", "upper.example", "localhost", "127.0.0.1",
                    "host.docker.internal", "postgres", "api.notion.com",
                }
            (episode / "runtime.json").write_text(json.dumps(runtime))
        elif args[0] == "exec" and "cat" in args:
            output = '{"pass": true}'
        return CompletedProcess(args, 0, stdout=output, stderr="")

    monkeypatch.setattr(gym, "_docker", docker)
    monkeypatch.setattr(gym, "_postgres_environment", lambda _: {})
    cleanup = MagicMock()
    monkeypatch.setattr(gym, "_cleanup_episode", cleanup)
    response = MagicMock()
    response.__enter__.return_value.status = 200
    monkeypatch.setattr(gym.urllib.request, "urlopen", lambda *args, **kwargs: response)
    error = RuntimeError("model failed") if agent_failed else None
    state = {
        "messages": [{"type": "ai", "content": "Answer"}],
        "decomposer_agent_runs": [{"status": "error" if agent_failed else "responded"}],
    }
    if skipped:
        state.update(stop_reason="sequence_limit", sequence_length=33043)
    monkeypatch.setattr(gym, "invoke_and_capture", AsyncMock(return_value=(state, error)))
    evaluation_saved_before_render = []

    def fail_render(trace):
        evaluation_saved_before_render.append(result_path.is_file())
        raise ValueError("broken renderer")

    monkeypatch.setattr(visualization, "render_trace", fail_render)
    args = gym.create_parser().parse_args([
        "task", "--episode-id", "episode", "--artifacts-dir", str(tmp_path / "traces"),
        "--evals-dir", str(tmp_path / "evals"),
    ])
    if agent_failed:
        with pytest.raises(RuntimeError, match="model failed") as caught:
            gym.run_episode(args)
        assert caught.value.__cause__ is error
    else:
        gym.run_episode(args)
    assert evaluation_saved_before_render == [True]
    assert "broken renderer" in caplog.text
    evaluation = json.loads(result_path.read_text())
    assert evaluation["native_pass"] is True
    assert evaluation["pass"] == (not agent_failed and not skipped)
    if skipped:
        assert evaluation["agent_error"] is None
        assert evaluation["stop_reason"] == "sequence_limit"
    trace = json.loads((episode / "trace.json").read_text())
    assert trace["model"] == trace["agent_model_id"] == "container/model"
    assert trace["agent_base_url"] == "http://image-provider/v1"
    assert trace["decomposer_generation_config"] == {"temperature": 0.3}
    assert (episode / "usage.json").is_file()
    assert (episode / "answer.txt").read_text() == "Answer"
    cleanup.assert_called_once()


@pytest.mark.parametrize("model_id", [
    "lmrouter/qwen_3_8_flash_next_low_thinking", "openrouter/qwen_3_8_flash_next_low_thinking",
])
def test_container_model_metadata_has_sampling_without_credentials(monkeypatch, model_id):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_PROXY_MASTER_KEY", "test-secret")
    metadata = model_metadata(model_id)
    assert metadata["model_id"] == model_id
    assert metadata["api_model"]
    assert metadata["base_url"].startswith("https://")
    config = metadata["generation_config"]
    assert config["temperature"] == 1.0
    if model_id.startswith("lmrouter/"):
        assert config["reasoning_effort"] == "low"
        assert config["preserve_reasoning"] is True
    else:
        assert config["reasoning"] == {"effort": "low"}
    serialized = json.dumps(metadata)
    assert "test-secret" not in serialized
    assert "api_key" not in serialized
    assert "http_client" not in serialized


def test_minimal_saves_failed_trace_and_preserves_error(tmp_path, monkeypatch, caplog):
    active = False

    @asynccontextmanager
    async def server(config):
        nonlocal active
        active = True
        yield "http://server"
        active = False

    failure = RuntimeError("model failed")
    state = {"decomposer_agent_runs": [{"status": "error", "error": repr(failure)}]}

    async def capture(*args):
        assert active
        return state, failure

    def render(trace):
        raise ValueError("broken renderer")

    monkeypatch.setattr(minimal, "__file__", str(tmp_path / "run.py"))
    monkeypatch.setattr(minimal, "agent_server", server)
    monkeypatch.setattr(minimal, "invoke_and_capture", capture)
    monkeypatch.setattr(visualization, "render_trace", render)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(minimal.main())
    assert caught.value is failure
    assert json.loads((tmp_path / "trace.json").read_text()) == state
    assert "broken renderer" in caplog.text
    assert not active


@pytest.mark.parametrize("openrouter_key", [None, "test-openrouter-key"])
def test_sft_collection_does_not_require_router_credentials(tmp_path, monkeypatch, openrouter_key):
    from sft.toolathlon_gym import collection

    monkeypatch.delenv("LLM_PROXY_MASTER_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    if openrouter_key:
        monkeypatch.setenv("OPENROUTER_API_KEY", openrouter_key)
    (tmp_path / "tasks/finalpool/task").mkdir(parents=True)
    docker = MagicMock(side_effect=RuntimeError("image unavailable"))
    with pytest.raises(RuntimeError, match="image unavailable"):
        collection.main(
            ["--tasks", "task"], repo_root=tmp_path, toolathlon_root=tmp_path,
            default_artifacts_dir=tmp_path / "artifacts", default_image="test-image",
            docker=docker,
        )
    docker.assert_called_once_with("image", "inspect", "test-image")


def test_sft_collection_orders_episodes_by_dataset_pass(tmp_path):
    from sft.toolathlon_gym import collection

    args = collection.parse_args(
        ["--all", "-n", "2", "--episode-timeout", "900"],
        {"image": "test-image", "artifacts_dir": tmp_path},
    )
    manifest = collection.create_manifest("test-run", ["task-a", "task-b"], 2, args)
    assert [episode["key"] for episode in manifest["episodes"]] == [
        "task-a::rep-001", "task-b::rep-001",
        "task-a::rep-002", "task-b::rep-002",
    ]
    assert manifest["config"]["episode_timeout"] == 900


@pytest.mark.parametrize("network_error", ["Pool overlaps with other one on this address space", "permission denied"])
def test_episode_network_allocation_retries_only_overlaps(tmp_path, monkeypatch, network_error):
    monkeypatch.delenv("LLM_PROXY_UNIX_SOCKET", raising=False)
    monkeypatch.setattr(gym, "TOOLATHLON_ROOT", tmp_path / "toolathlon")
    (gym.TOOLATHLON_ROOT / "tasks/finalpool/task").mkdir(parents=True)
    subnets = []

    def docker(*args, check=True):
        if args[:2] == ("network", "create"):
            subnets.append(args[args.index("--subnet") + 1])
            if len(subnets) == 1:
                return CompletedProcess(args, 1, stdout="", stderr=network_error)
        if args[0] == "run":
            raise RuntimeError("reached PostgreSQL startup")
        return CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(gym, "_docker", docker)
    cleanup = MagicMock()
    monkeypatch.setattr(gym, "_cleanup_episode", cleanup)
    args = gym.create_parser().parse_args([
        "task", "--episode-id", "episode", "--artifacts-dir", str(tmp_path / "traces"),
        "--evals-dir", str(tmp_path / "evals"),
    ])
    overlap = network_error.startswith("Pool overlaps")
    with pytest.raises(RuntimeError, match="reached PostgreSQL startup" if overlap else "permission denied"):
        gym.run_episode(args)
    assert len(subnets) == (2 if overlap else 1)
    assert len(set(subnets)) == len(subnets)
    assert all(subnet.startswith("10.240.") and subnet.endswith(".0/24") for subnet in subnets)
    cleanup.assert_called_once()


def test_collection_eta_counts_all_outcomes_and_resets_on_resume(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from sft.toolathlon_gym import collection

    (tmp_path / "tasks/finalpool/task").mkdir(parents=True)
    clock = [0.0]
    monkeypatch.setattr(collection, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    outcomes = iter([
        ("completed", True), ("completed", False), ("skipped", None),
        ("failed", False), ("completed", None), ("completed", True),
    ])

    def execute(args, *, attempt, **kwargs):
        assert args.episode_timeout == 1800
        clock[0] += 60
        status, score = next(outcomes)
        error = {"message": "runner failed", "stderr_tail": "Traceback...\nRuntimeError: backend unavailable\n"} if status == "failed" else None
        return {"attempt": attempt, "status": status, "error": error,
                "score": score, "duration_seconds": 60}

    monkeypatch.setattr(collection, "execute_episode", execute)
    image = [{"Config": {"Labels": {"org.decomposer.sft.sequence-limit": "16384"}}}]
    kwargs = dict(repo_root=tmp_path, toolathlon_root=tmp_path,
                  default_artifacts_dir=tmp_path / "artifacts", default_image="test-image",
                  docker=lambda *args: CompletedProcess(args, 0, stdout=json.dumps(image)))
    manifest = collection.main(
        ["--tasks", "task", "-n", "5", "--concurrency", "1", "--episode-timeout", "1800"],
        **kwargs,
    )
    output = capsys.readouterr().out
    assert "1/5 finished | solved=1 unsolved=0 unscored=0 errors=0 length_limit=0" in output
    assert "60.0 episodes/hour | remaining ~4m" in output
    assert "solved=1 unsolved=1 unscored=1 errors=1 length_limit=1" in output
    assert "remaining ~unreliable (episode errors)" in output
    assert "ERRORS task::rep-004 attempt=1 duration=1m | RuntimeError: backend unavailable | logs:" in output
    assert "LENGTH_LIMIT task::rep-003" in output
    assert "UNSOLVED task::rep-002" in output
    assert "UNSCORED task::rep-005" in output
    assert "active=0 queued=0 | elapsed 5m | 60.0 episodes/hour | remaining ~0s" in output
    clock[0] += 3600
    collection.main(["--resume", manifest["run_id"]], **kwargs)
    output = capsys.readouterr().out
    assert "1/1 finished | solved=1 unsolved=0 unscored=0 errors=0 length_limit=0" in output
    assert "elapsed 1m | 60.0 episodes/hour | remaining ~0s" in output


def test_progress_duration_includes_days():
    from sft.toolathlon_gym.collection import format_duration

    assert format_duration(90061) == "1d 1h 1m 1s"
    assert format_duration(0) == "0s"


def test_worker_prompt_uses_workspace_and_excludes_completion_tools(monkeypatch):
    import sys
    from types import SimpleNamespace
    from gyms.toolathlon_gym import agents, tools

    available = [SimpleNamespace(name=name) for name in
                 ("python_execute", "claim_done", "local-claim_done")]
    monkeypatch.setattr(tools.app.state, "tools", available, raising=False)
    monkeypatch.setattr(tools.app.state, "workspace", "/workspace/agent", raising=False)
    monkeypatch.setitem(sys.modules, "tools", tools)
    monkeypatch.setitem(sys.modules, "model_logging", SimpleNamespace(durable_model_call_log=object()))
    monkeypatch.setattr(agents, "create_model", lambda _: object())
    factory = MagicMock()
    monkeypatch.setattr(agents, "create_agent", factory)
    agents.qwen_3_5_4b_thinking()
    kwargs = factory.call_args.kwargs
    assert kwargs["tools"] == [available[0]]
    assert tools.app.state.tools == available
    assert "`/workspace/agent`" in kwargs["system_prompt"]
    assert "claim_done" not in kwargs["system_prompt"]
    assert "Further progress requires tools, access, information, or a decision" in kwargs["system_prompt"]
