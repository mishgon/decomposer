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


@pytest.mark.parametrize("agent_failed", [False, True])
def test_render_failure_preserves_evaluation_and_container_metadata(tmp_path, monkeypatch, caplog, agent_failed):
    monkeypatch.delenv("LLM_PROXY_MASTER_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-secret")
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
    assert evaluation["pass"] is not agent_failed
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
