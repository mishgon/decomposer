import asyncio
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
from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from gyms.toolathlon import batch, mlspace_serve, run, settings, teacher_models, usage


def test_evaluation_runtime_settings_are_uniform() -> None:
    assert settings.SUBAGENT_CONTEXT_TOKENS == 256_000
    assert settings.SUBAGENT_RECURSION_LIMIT == 410
    assert settings.DECOMPOSER_RECURSION_LIMIT == 410
    assert settings.DEEPSEEK_REASONING_EFFORT == "high"


def test_batch_rejects_more_than_one_gpu_per_model() -> None:
    defaults = {
        "model": "teacher",
        "subagent_model": "subagent",
        "subagent_port": 8030,
        "image": "image",
        "artifacts_dir": Path("artifacts/gyms/toolathlon"),
    }
    with pytest.raises(SystemExit):
        batch.parse_args(
            [
                "--tasks",
                "finalpool/example",
                "--purpose",
                "evaluation",
                "--subagent-gpu",
                "0,1",
                "--vllm-data-parallel-size",
                "2",
            ],
            defaults,
        )


def test_evaluation_metrics_report_pass_at_and_pass_power_three() -> None:
    manifest = {
        "config": {"repetitions": 3},
        "episodes": [
            {"task": "a", "score": score} for score in (True, False, False)
        ]
        + [{"task": "b", "score": score} for score in (True, True, True)],
    }

    metrics = batch.evaluation_metrics(manifest)

    assert metrics["pass@1"] == pytest.approx(4 / 6)
    assert metrics["pass@3"] == 1.0
    assert metrics["pass^3"] == 0.5
    assert metrics["scored_task_count"] == 2
    assert metrics["unscored_trials"] == 0


def test_evaluation_metrics_do_not_hide_unscored_tasks() -> None:
    manifest = {
        "config": {"repetitions": 3},
        "episodes": [
            {"task": "a", "score": score} for score in (True, False, None)
        ],
    }

    metrics = batch.evaluation_metrics(manifest)

    assert metrics["pass@1"] is None
    assert metrics["pass@3"] is None
    assert metrics["pass^3"] is None
    assert metrics["scored_task_count"] == 0
    assert metrics["unscored_trials"] == 1


def test_evaluation_metrics_do_not_drop_one_incomplete_task() -> None:
    manifest = {
        "config": {"repetitions": 3},
        "episodes": [
            {"task": "complete", "score": score}
            for score in (True, True, True)
        ]
        + [
            {"task": "incomplete", "score": score}
            for score in (False, False, None)
        ],
    }

    metrics = batch.evaluation_metrics(manifest)

    assert metrics["scored_task_count"] == 1
    assert metrics["unscored_trials"] == 1
    assert metrics["pass@1"] is None
    assert metrics["pass@3"] is None
    assert metrics["pass^3"] is None


def test_all_valid_metrics_use_full_benchmark_denominator() -> None:
    manifest = {
        "config": {
            "repetitions": 3,
            "benchmark_task_count": 108,
            "unrun_tasks_are_failures": True,
        },
        "episodes": [
            {"task": "a", "score": score} for score in (True, False, False)
        ]
        + [{"task": "b", "score": score} for score in (True, True, True)],
    }

    metrics = batch.evaluation_metrics(manifest)

    assert metrics["task_count"] == 2
    assert metrics["benchmark_task_count"] == 108
    assert metrics["unrun_task_count"] == 106
    assert metrics["assumed_failed_task_count"] == 106
    assert metrics["expected_trials"] == 324
    assert metrics["pass@1"] == pytest.approx(4 / 324)
    assert metrics["pass@3"] == pytest.approx(2 / 108)
    assert metrics["pass^3"] == pytest.approx(1 / 108)


from gyms.toolathlon.subagents import graph as subagent_graph
from gyms.toolathlon.subagents.model_logging import durable_model_call_log
from gyms.toolathlon.subagents.openrouter_compat import create_openrouter_model
from gyms.toolathlon.subagents.webapp import truncate_mcp_tool_output


def test_configured_subagents_are_registered() -> None:
    registered = json.loads(
        (Path(run.__file__).parent / "subagents" / "langgraph.json").read_text()
    )["graphs"]

    assert {
        assistant_id for _, assistant_id, _ in run.SUBAGENT_TYPES
    } <= registered.keys()
    assert "deepseek_openrouter" in registered
    assert [item[0] for item in run.SUBAGENT_TYPES] == [
        "qwen_3_5_4b_non_thinking",
        "gemma_4_e4b_thinking",
        "gemma_4_26b_a4b_non_thinking",
    ]


def test_decomposer_prompt_modes_select_student_and_teacher_prompts() -> None:
    assert run.DECOMPOSER_PROMPTS["student"] == run.DECOMPOSER_SYSTEM_PROMPT
    assert (
        run.DECOMPOSER_PROMPTS["teacher"]
        == run.DECOMPOSER_TEACHER_SYSTEM_PROMPT
    )


def test_agent_failure_keeps_artifact_grade_out_of_official_score() -> None:
    official, artifact = run.evaluation_scores(
        {"pass": True}, agent_success=False
    )

    assert official is None
    assert artifact is True


def test_successful_agent_uses_native_grade_as_official_score() -> None:
    official, artifact = run.evaluation_scores(
        {"pass": False}, agent_success=True
    )

    assert official is False
    assert artifact is False


def test_subagent_reconnects_model_requests_for_data_parallel_balance(
    monkeypatch,
) -> None:
    captured = {}
    compiled = object()

    class FakeChatVLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(subagent_graph, "ChatVLLM", FakeChatVLLM)
    monkeypatch.setattr(subagent_graph, "get_tools", lambda: [])
    monkeypatch.setattr(
        subagent_graph, "create_agent", lambda **_kwargs: compiled
    )

    result = subagent_graph._create_subagent(
        "model", "UNSET_TEST_BASE_URL", 9999, thinking=False
    )

    client = captured["http_async_client"]
    assert client._transport._pool._max_keepalive_connections == 0
    assert "max_tokens" not in captured
    asyncio.run(client.aclose())
    assert result is compiled


def test_subagent_uses_task_specific_system_prompt(monkeypatch) -> None:
    model_kwargs = {}
    agent_kwargs = {}

    class FakeChatVLLM:
        def __init__(self, **kwargs):
            model_kwargs.update(kwargs)

    monkeypatch.setattr(subagent_graph, "ChatVLLM", FakeChatVLLM)
    monkeypatch.setattr(subagent_graph, "get_tools", lambda: [])
    monkeypatch.setattr(
        subagent_graph,
        "create_agent",
        lambda **kwargs: agent_kwargs.update(kwargs) or object(),
    )

    subagent_graph._create_subagent(
        "model",
        "UNSET_TEST_BASE_URL",
        9999,
        thinking=False,
        system_prompt="benchmark-specific instructions",
    )

    assert agent_kwargs["system_prompt"] == "benchmark-specific instructions"
    asyncio.run(model_kwargs["http_async_client"].aclose())


def test_tool_output_limit_matches_toolathlon_default() -> None:
    from gyms.toolathlon.subagents import webapp

    assert webapp.MAX_TOOL_OUTPUT_CHARS == 100_000


def test_qwen_non_thinking_uses_official_sampling_parameters(monkeypatch) -> None:
    captured = {}
    compiled = object()

    class FakeChatVLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(subagent_graph, "ChatVLLM", FakeChatVLLM)
    monkeypatch.setattr(subagent_graph, "get_tools", lambda: [])
    monkeypatch.setattr(
        subagent_graph, "create_agent", lambda **_kwargs: compiled
    )

    result = subagent_graph._create_subagent(
        "Qwen/Qwen3.5-4B", "UNSET_TEST_BASE_URL", 9999, thinking=False
    )

    assert captured["temperature"] == 0.7
    assert captured["top_p"] == 0.8
    assert captured["presence_penalty"] == 1.5
    assert captured["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "include_reasoning": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    asyncio.run(captured["http_async_client"].aclose())
    assert result is compiled


def test_gemma_non_thinking_uses_official_sampling_parameters(monkeypatch) -> None:
    captured = {}
    compiled = object()

    class FakeChatVLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(subagent_graph, "ChatVLLM", FakeChatVLLM)
    monkeypatch.setattr(subagent_graph, "get_tools", lambda: [])
    monkeypatch.setattr(
        subagent_graph, "create_agent", lambda **_kwargs: compiled
    )

    result = subagent_graph.gemma_4_26b_a4b_non_thinking()

    assert captured["model"] == "google/gemma-4-26B-A4B-it"
    assert captured["temperature"] == 1.0
    assert captured["top_p"] == 0.95
    assert captured["extra_body"] == {
        "top_k": 64,
        "include_reasoning": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert captured["preserve_reasoning"] is False
    asyncio.run(captured["http_async_client"].aclose())
    assert result is compiled


def test_gemma_31b_thinking_uses_official_sampling_parameters(monkeypatch) -> None:
    captured = {}
    compiled = object()

    class FakeChatVLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(subagent_graph, "ChatVLLM", FakeChatVLLM)
    monkeypatch.setattr(subagent_graph, "get_tools", lambda: [])
    monkeypatch.setattr(
        subagent_graph, "create_agent", lambda **_kwargs: compiled
    )

    result = subagent_graph.gemma_4_31b_thinking()

    assert captured["model"] == "google/gemma-4-31B-it"
    assert captured["temperature"] == 1.0
    assert captured["top_p"] == 0.95
    assert captured["extra_body"] == {
        "top_k": 64,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert captured["preserve_reasoning"] is True
    asyncio.run(captured["http_async_client"].aclose())
    assert result is compiled


def test_deepseek_subagent_uses_configured_openrouter_model(monkeypatch) -> None:
    captured = {}
    compiled = object()

    def fake_create_openrouter_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("TOOLATHLON_OPENROUTER_MODEL", "deepseek/test")
    monkeypatch.setattr(
        subagent_graph, "create_openrouter_model", fake_create_openrouter_model
    )
    monkeypatch.setattr(subagent_graph, "get_tools", lambda: [])
    monkeypatch.setattr(
        subagent_graph, "create_agent", lambda **_kwargs: compiled
    )

    result = subagent_graph.deepseek_openrouter()

    assert captured["model"] == "deepseek/test"
    assert "max_tokens" not in captured
    assert captured["reasoning"] == {"effort": "high"}
    assert result is compiled


def test_openrouter_model_accepts_local_relay_base_url(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv(
        "TOOLATHLON_OPENROUTER_BASE_URL", "http://127.0.0.1:18041/api/v1"
    )

    model = create_openrouter_model(
        model="deepseek/test",
        reasoning={"effort": "high"},
        max_tokens=8,
        timeout=1,
        max_retries=0,
    )

    assert str(model.openai_api_base) == "http://127.0.0.1:18041/api/v1"


def test_lmrouter_teacher_is_thinking(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROXY_URL", "https://lmrouter.example/v1")
    monkeypatch.setenv("LLM_PROXY_MASTER_KEY", "secret")

    model = teacher_models.create_lmrouter_teacher(
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        timeout=180,
        max_retries=5,
    )

    assert model.model_name == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert model.extra_body["chat_template_kwargs"] == {"enable_thinking": True}
    assert model.extra_body["include_reasoning"] is True
    assert model.preserve_reasoning is True
    assert model.parse_qwen_xml_tool_calls is True
    assert model.max_tokens is None


def test_usage_summary_separates_decomposer_and_subagents() -> None:
    def message(input_tokens, output_tokens, *, cache=0, reasoning=0, cost=None):
        return {
            "type": "ai",
            "data": {
                "usage_metadata": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "input_token_details": {"cache_read": cache},
                    "output_token_details": {"reasoning": reasoning},
                },
                "response_metadata": {"token_usage": {"cost": cost}},
            },
        }

    summary = usage.build_usage_summary(
        [message(100, 10, cache=40, reasoning=3, cost=0.01)],
        {
            "sub-1": {
                "subagent_type_id": "qwen",
                "status": "success",
                "messages": [message(200, 20), {"type": "tool", "content": "ok"}],
            }
        },
    )

    assert summary["decomposer"]["input_tokens"] == 100
    assert summary["subagents"]["sub-1"]["output_tokens"] == 20
    assert summary["totals"]["total_tokens"] == 330
    assert summary["totals"]["cache_read_tokens"] == 40
    assert summary["totals"]["reasoning_tokens"] == 3
    assert summary["totals"]["cost"] == pytest.approx(0.01)


def test_subagent_model_calls_are_durably_logged(tmp_path, monkeypatch) -> None:
    path = tmp_path / "calls.jsonl"
    monkeypatch.setenv("TOOLATHLON_SUBAGENT_CALL_LOG", str(path))
    request = SimpleNamespace(
        model=SimpleNamespace(model_name="test-model"),
        messages=[HumanMessage(content="do it")],
        system_message=None,
    )

    async def handler(_request):
        return ModelResponse(
            result=[
                AIMessage(
                    content="done",
                    usage_metadata={
                        "input_tokens": 3,
                        "output_tokens": 1,
                        "total_tokens": 4,
                    },
                )
            ]
        )

    asyncio.run(durable_model_call_log.awrap_model_call(request, handler))
    record = json.loads(path.read_text())
    assert record["status"] == "success"
    assert record["model"] == "test-model"
    assert record["request_delta"][0]["data"]["content"] == "do it"
    assert record["response"][0]["data"]["usage_metadata"]["total_tokens"] == 4


def test_subagent_turns_mcp_failures_into_tool_errors() -> None:
    async def failing_handler(_request):
        raise RuntimeError("ping")

    request = SimpleNamespace(tool_call={"id": "call-1"})
    response = asyncio.run(
        truncate_mcp_tool_output.awrap_tool_call(request, failing_handler)
    )

    assert response.tool_call_id == "call-1"
    assert response.status == "error"
    assert response.content == "Tool call failed: ping"


def test_overlong_tool_output_is_preserved_in_workspace(
    tmp_path, monkeypatch
) -> None:
    from gyms.toolathlon.subagents import webapp

    monkeypatch.setattr(webapp, "MAX_TOOL_OUTPUT_CHARS", 10)
    monkeypatch.setenv("TOOLATHLON_AGENT_WORKSPACE", str(tmp_path))
    request = SimpleNamespace(tool_call={"id": "call-1"})

    async def handler(_request):
        return ToolMessage(
            content=[{"type": "text", "text": "x" * 20}],
            tool_call_id="call-1",
        )

    response = asyncio.run(
        webapp.truncate_mcp_tool_output.awrap_tool_call(request, handler)
    )

    output_files = list((tmp_path / ".overlong_tool_outputs").glob("*.json"))
    assert len(output_files) == 1
    assert json.loads(output_files[0].read_text()) == [
        {"type": "text", "text": "x" * 20}
    ]
    assert f"shortuuid identifier {output_files[0].stem}" in response.content
    assert str(output_files[0].relative_to(tmp_path)) in response.content


def test_docker(monkeypatch) -> None:
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "output", "")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    result = run._docker("ps", check=False)

    assert result.stdout == "output"
    assert calls == [
        (
            (["docker", "ps"],),
            {
                "capture_output": True,
                "text": True,
                "timeout": run.DOCKER_COMMAND_TIMEOUT,
            },
        )
    ]

    calls.clear()
    run._docker("exec", "container", "true")
    assert calls[0][1]["timeout"] == run.DOCKER_EXEC_TIMEOUT


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


def test_docker_timeout_names_the_command(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="docker inspect x timed out after 60s"):
        run._docker("inspect", "x")


def test_copy_user_configs_uses_absolute_container_destinations(
    tmp_path, monkeypatch
) -> None:
    config = tmp_path / "configs" / "global_configs.py"
    config.parent.mkdir()
    config.write_text("global_configs = {}\n")
    shortnames = tmp_path / "podman-shortnames.conf"
    shortnames.write_text('[aliases]\n"mysql" = "docker.io/library/mysql"\n')
    calls = []
    monkeypatch.setattr(run, "TOOLATHLON_ROOT", tmp_path)
    monkeypatch.setattr(run, "PODMAN_SHORTNAMES_FILE", shortnames)
    monkeypatch.setattr(run, "USER_CONFIG_FILES", ("configs/global_configs.py",))
    monkeypatch.setattr(run, "_docker", lambda *args, **kwargs: calls.append(args))

    run._copy_user_configs("task-container")

    assert calls == [
        (
            "cp",
            str(shortnames),
            "task-container:/workspace/configs/podman-shortnames.conf",
        ),
        (
            "cp",
            str(config),
            "task-container:/workspace/configs/global_configs.py",
        )
    ]


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

    data_parallel_command = run.vllm_command(
        "/models/qwen",
        8030,
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


def test_served_subagent_model_names_cover_supported_local_models() -> None:
    assert run.served_subagent_model_name("/models/Qwen3.5-4B") == (
        "Qwen/Qwen3.5-4B"
    )
    assert run.served_subagent_model_name("/models/gemma-4-E4B-it") == (
        "google/gemma-4-E4B-it"
    )
    assert run.served_subagent_model_name("/models/gemma-4-26B-A4B-it") == (
        "google/gemma-4-26B-A4B-it"
    )
    assert run.served_subagent_model_name("/models/gemma-4-31B-it") == (
        "google/gemma-4-31B-it"
    )


def test_official_simple_agent_bundle_uses_local_vllm_and_native_paths() -> None:
    source = {
        "container_paths": {
            "task_root": "/workspace/dumps",
            "agent_workspace": "/workspace/dumps/workspace",
            "log_file": "/workspace/dumps/traj_log.json",
        },
        "host_paths": {
            "task_root": "/host/episode",
            "agent_workspace": "/host/episode/workspace",
            "log_file": "/host/episode/traj_log.json",
        },
        "eval_config": {
            "global_task_config": {"max_steps_under_single_turn_mode": 1},
            "agent": {},
        },
    }
    args = SimpleNamespace(
        subagent_model="/models/gemma-4-31B-it",
        vllm_max_model_len=131072,
        max_steps=200,
    )

    result = run.official_simple_agent_bundle(source, args)

    assert result["host_paths"] == source["container_paths"]
    assert source["host_paths"]["task_root"] == "/host/episode"
    agent = result["eval_config"]["agent"]
    assert agent["model"] == {
        "short_name": "google/gemma-4-31B-it",
        "provider": "local_vllm",
        "context_window": 131072,
    }
    assert agent["generation"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": None,
        "extra_body": {
            "top_k": 64,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }
    assert agent["tool"]["parallel_tool_calls"] is True
    assert agent["tool"]["max_inner_turns"] == 200


def test_answer_from_native_trajectory_uses_last_assistant_message() -> None:
    assert run.answer_from_native_trajectory(
        {
            "messages": [
                {"role": "assistant", "content": "earlier"},
                {"role": "tool", "content": "result"},
                {"role": "assistant", "content": "done"},
            ]
        }
    ) == "done"


def test_vllm_command_uses_gemma_thinking_parsers() -> None:
    command = run.vllm_command(
        "/models/gemma-4-E4B-it",
        8030,
        max_model_len=256000,
        gpu_memory_utilization=0.9,
    )

    assert command[command.index("--served-model-name") + 1] == (
        "google/gemma-4-E4B-it"
    )
    assert command[command.index("--tool-call-parser") + 1] == "gemma4"
    assert command[command.index("--reasoning-parser") + 1] == "gemma4"
    assert "--enable-prefix-caching" in command
    assert command[command.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking":true}'
    )


def test_vllm_command_uses_gemma_31b_thinking() -> None:
    command = run.vllm_command(
        "/models/gemma-4-31B-it",
        8030,
        max_model_len=131072,
        gpu_memory_utilization=0.9,
    )

    assert command[command.index("--served-model-name") + 1] == (
        "google/gemma-4-31B-it"
    )
    assert command[command.index("--tool-call-parser") + 1] == "gemma4"
    assert command[command.index("--reasoning-parser") + 1] == "gemma4"
    assert command[command.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking":true}'
    )


def test_vllm_command_uses_gemma_26b_non_thinking() -> None:
    command = run.vllm_command(
        "/models/gemma-4-26B-A4B-it",
        8030,
        max_model_len=256000,
        gpu_memory_utilization=0.9,
    )

    assert command[command.index("--served-model-name") + 1] == (
        "google/gemma-4-26B-A4B-it"
    )
    assert command[command.index("--tool-call-parser") + 1] == "gemma4"
    assert command[command.index("--reasoning-parser") + 1] == "gemma4"
    assert command[command.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking":false}'
    )


def test_start_vllm_refuses_an_occupied_port(tmp_path) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    try:
        with pytest.raises(RuntimeError, match=f"port {port} is already in use"):
            run.start_vllm(
                model="model",
                port=port,
                gpu="0",
                max_model_len=1024,
                gpu_memory_utilization=0.5,
                timeout=1,
                log_path=tmp_path / "vllm.log",
                reuse=False,
            )
    finally:
        listener.close()


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
    assert run.resolve_docker_socket("/var/run/docker.sock") == "/var/run/docker.sock"


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


def test_resolve_docker_socket_prefers_rootless_over_system_podman(
    tmp_path, monkeypatch
) -> None:
    xdg_dir = tmp_path / "xdg"
    rootless_socket = xdg_dir / "podman" / "podman.sock"
    rootless_socket.parent.mkdir(parents=True)
    rootless_socket.touch()
    system_socket = tmp_path / "system-podman.sock"
    system_socket.touch()
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg_dir))
    monkeypatch.setattr(run, "DOCKER_SOCKET_CANDIDATES", (str(system_socket),))

    assert run.resolve_docker_socket() == str(rootless_socket)


def test_resolve_docker_socket_fails_when_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(
        run, "DOCKER_SOCKET_CANDIDATES", (str(tmp_path / "missing.sock"),)
    )

    with pytest.raises(RuntimeError, match="No Docker socket found"):
        run.resolve_docker_socket()


def test_container_socket_is_available_to_docker_and_podman() -> None:
    assert run.container_socket_mounts("/run/user/123/podman/podman.sock") == [
        "-v",
        "/run/user/123/podman/podman.sock:/var/run/docker.sock",
        "-v",
        "/run/user/123/podman/podman.sock:/run/podman/podman.sock",
    ]


def test_k8s_tasks_have_post_evaluation_cluster_cleanup() -> None:
    assert set(run.K8S_TASK_CLEANUP_COMMANDS) == {
        "finalpool/k8s-deployment-cleanup",
        "finalpool/k8s-mysql",
        "finalpool/k8s-pr-preview-testing",
        "finalpool/k8s-redis-helm-upgrade",
        "finalpool/k8s-safety-audit",
    }
    assert run.K8S_TASK_CLEANUP_COMMANDS["finalpool/k8s-pr-preview-testing"][-2:] == (
        "_",
        "stop",
    )


def test_shared_mutable_services_are_serialized_across_tasks() -> None:
    assert batch.shared_task_resources("finalpool/canvas-art-manager") == (
        "canvas",
    )
    assert batch.shared_task_resources("finalpool/k8s-mysql") == ("k8s",)
    assert batch.shared_task_resources("finalpool/meeting-assign") == ()


def test_episode_command_passes_docker_socket(tmp_path) -> None:
    args = SimpleNamespace(
        purpose="evaluation",
        agent_mode="simple",
        decomposer_prompt="student",
        model="model",
        subagent_model="subagent-model",
        subagent_port=8030,
        subagent_gpu="0",
        vllm_max_model_len=65536,
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
    assert command[command.index("--agent-mode") + 1] == "simple"
    assert command[command.index("--agent-system-prompt") + 1] == "toolathlon"
    assert command[command.index("--max-tool-output-chars") + 1] == "100000"
    assert command[command.index("--decomposer-prompt") + 1] == "student"

    args.container_slots = 2
    slotted = batch.episode_command(
        args,
        Path("run.py"),
        "finalpool/example",
        "run-1",
        1,
        1,
        "ep-1",
        tmp_path,
        container_slot=1,
    )
    assert slotted[slotted.index("--container-lock-file") + 1].endswith(
        "container-01.lock"
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

    assert batch.select_tasks(
        tmp_path, run_all=True, run_all_valid=False, requested=None
    ) == [
        "finalpool/alpha",
        "finalpool/beta",
    ]
    assert batch.select_tasks(
        tmp_path,
        run_all=False,
        run_all_valid=False,
        requested=["finalpool/beta", "finalpool/alpha"],
    ) == ["finalpool/beta", "finalpool/alpha"]
    with pytest.raises(ValueError, match="Unknown Toolathlon task"):
        batch.select_tasks(
            tmp_path,
            run_all=False,
            run_all_valid=False,
            requested=["finalpool/missing"],
        )

    assert batch.wants_batch(["--repetitions=2"])
    assert batch.wants_batch(["-n2"])
    assert batch.wants_batch(["--all-valid"])


def test_all_valid_suite_has_55_tasks_and_excludes_known_infra_failures() -> None:
    assert len(batch.VALID_EVALUATION_TASKS) == 55
    assert "finalpool/fillout-online-forms" not in batch.VALID_EVALUATION_TASKS
    assert "finalpool/git-milestone" not in batch.VALID_EVALUATION_TASKS


def test_batch_manifest_spreads_repetitions_across_rounds() -> None:
    defaults = {
        "model": "decomposer-model",
        "subagent_model": "subagent-model",
        "subagent_port": 8000,
        "image": "image",
        "artifacts_dir": Path("artifacts"),
    }
    args = batch.parse_args(
        [
            "--tasks",
            "finalpool/alpha",
            "finalpool/beta",
            "--purpose",
            "evaluation",
            "-n",
            "3",
        ],
        defaults,
    )

    manifest = batch.create_manifest(
        "run-1", ["finalpool/alpha", "finalpool/beta"], 3, args
    )

    assert [episode["key"] for episode in manifest["episodes"]] == [
        "finalpool/alpha::rep-001",
        "finalpool/beta::rep-001",
        "finalpool/alpha::rep-002",
        "finalpool/beta::rep-002",
        "finalpool/alpha::rep-003",
        "finalpool/beta::rep-003",
    ]


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
    assert episode_calls[0]["episode"]["task"] == "finalpool/beta"
    assert episode_calls[0]["episode"]["repetition"] == 1
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
            "--container-slots",
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


def test_batch_serializes_shared_resources_without_blocking_worker_slots(
    tmp_path, monkeypatch
) -> None:
    toolathlon_root = tmp_path / "toolathlon"
    tasks = [
        "finalpool/canvas-alpha",
        "finalpool/canvas-beta",
        "finalpool/independent",
    ]
    for task in tasks:
        (toolathlon_root / "tasks" / task).mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(batch, "new_run_id", lambda: "resource-aware-run")

    canvas_alpha_started = threading.Event()
    independent_started = threading.Event()
    canvas_alpha_finished = threading.Event()
    canvas_beta_started_early = False

    def fake_execute_episode(args, **kwargs):
        nonlocal canvas_beta_started_early
        task = kwargs["episode"]["task"]
        if task == "finalpool/canvas-alpha":
            canvas_alpha_started.set()
            independent_started.wait(timeout=1)
            time.sleep(0.02)
            canvas_alpha_finished.set()
        elif task == "finalpool/independent":
            canvas_alpha_started.wait(timeout=1)
            independent_started.set()
        elif task == "finalpool/canvas-beta":
            canvas_beta_started_early = not canvas_alpha_finished.is_set()
        attempt = kwargs["attempt"]
        return {
            "attempt": attempt,
            "status": "completed",
            "score": True,
            "artifact_path": "/trace",
            "evaluation_path": "/eval/result.json",
            "started_at": "start",
            "finished_at": "finish",
            "duration_seconds": 0.02,
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
            "2",
            "--container-slots",
            "2",
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

    assert independent_started.is_set()
    assert canvas_beta_started_early is False
    assert manifest["counts"]["completed"] == 3


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
            "--container-slots",
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

    assert ("rm", "--force", "--time", "0", "task-container") in calls
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
        agent_mode="simple",
        model="decomposer-model",
        subagent_model="subagent-model",
        subagent_port=8030,
        subagent_gpu="1",
        vllm_max_model_len=32768,
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
    assert result["artifact_score"] is False
    assert result["artifact_path"] == str(artifact_dir)
    assert result["evaluation_path"] == str(evaluation_path)
    command = popen_calls[0][0]
    assert command[command.index("--episode-id") + 1] == episode_id
    assert command[command.index("--run-id") + 1] == "run-id"
    assert command[command.index("--n-jobs-per-worker") + 1] == "1000"
    assert command[command.index("--stashes-dir") + 1] == str(root / "stashes")
    assert command[command.index("--container-lock-file") + 1] == str(
        root / "runs" / "run-id" / "container.lock"
    )
