from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from gyms.tau2_gym import run as run_module
from gyms.tau2_gym.experiments import (
    EXPERIMENTS,
    SUBAGENT_DESCRIPTION,
    SUBAGENT_TYPE_ID,
    Tau2Experiment,
    get_experiment,
)
from gyms.tau2_gym.task_pools import load_pool
from gyms.qwen_sampling import QWEN35_GENERAL_NON_THINKING

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_experiment_names_are_unique_and_pools_exist() -> None:
    names = [experiment.name for experiment in EXPERIMENTS]
    assert len(names) == len(set(names))
    for experiment in EXPERIMENTS:
        load_pool(experiment.pool)


def test_subagent_schema_matches_the_sft_releases() -> None:
    """Teacher traces, SFT data, evals and OPD must share one subagent type and tool schema."""
    spec = yaml.safe_load(
        (REPO_ROOT / "sft/specs/decomposer_mixed_deepseek_qwen35_4b_nonthinking_v5_32k_student.yaml").read_text()
    )
    (subagent,) = spec["policy"]["subagent_types"]
    assert subagent == {"id": SUBAGENT_TYPE_ID, "description": SUBAGENT_DESCRIPTION}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"manager_backend": "llm_proxy", "manager_checkpoint": Path("/x")}, "local_vllm"),
        ({"manager_backend": "llm_proxy", "return_token_ids": True}, "local_vllm"),
        ({"manager_backend": "llm_proxy", "manager_extra_body": {"a": 1}}, "openrouter only"),
        ({"manager_backend": "llm_proxy", "manager_sampling": None}, "needs manager_sampling"),
        ({"manager_reasoning_mode": "thinking"}, "non_thinking"),
        ({"prompt_profile": "expert"}, "unknown prompt profile"),
        ({"concurrency": 0}, "at least 1"),
        # effort only means something to a thinking manager behind the proxy
        ({"manager_reasoning_effort": "low"}, "llm_proxy thinking manager"),
        ({"manager_backend": "llm_proxy", "manager_reasoning_effort": "low"}, "llm_proxy thinking manager"),
    ],
)
def test_invalid_experiments_are_rejected(overrides: dict, message: str) -> None:
    base = dict(
        name="x",
        description="x",
        manager_backend="local_vllm",
        manager_model_id="decomposer/x",
        prompt_profile="student",
        pool="decomposer_eval_v1",
        manager_reasoning_mode="non_thinking",
        manager_sampling=QWEN35_GENERAL_NON_THINKING,
    )
    with pytest.raises(ValueError, match=message):
        Tau2Experiment(**(base | overrides))


def _policy(config: dict) -> tuple[str, dict]:
    ((name, value),) = config["policy_model"]["responses_api_models"].items()
    return name, value


def test_local_manager_config_records_token_ids_only_for_opd() -> None:
    ports = run_module.PortLayout(offset=100).shifted()
    opd = run_module.gym_config(get_experiment("opd_rollout"), ports)
    name, policy = _policy(opd)
    assert name == "vllm_model"
    assert policy["base_url"] == "http://127.0.0.1:8126/v1"
    assert policy["return_token_id_information"] is True
    # OPD samples the raw policy: untruncated, unpenalised.
    assert opd["responses_create_params"] == {"temperature": 1.0, "top_p": 1.0, "presence_penalty": 0.0}
    assert policy["extra_body"]["top_k"] == -1

    _, evaluation = _policy(run_module.gym_config(get_experiment("qwen35_4b_sft_mixed_v3_student"), ports))
    assert evaluation["return_token_id_information"] is False


def test_remote_manager_configs() -> None:
    ports = run_module.PortLayout()
    name, policy = _policy(run_module.gym_config(get_experiment("qwen38_flash_teacher_thinking"), ports))
    assert name == "openai_model"
    assert policy["openai_base_url"] == "http://127.0.0.1:8144/v1"

    name, policy = _policy(run_module.gym_config(get_experiment("deepseek_v4_flash_teacher"), ports))
    assert policy["openai_api_key"] == "${oc.env:OPENROUTER_API_KEY_DECOMPOSER}"
    assert policy["extra_body"] == {"reasoning": {"effort": "max"}}


def test_agent_block_follows_the_experiment_and_port_offset() -> None:
    ports = run_module.PortLayout(offset=7).shifted()
    config = run_module.gym_config(get_experiment("qwen38_flash_teacher_non_thinking"), ports)
    agent = config["decomposer"]["responses_api_agents"]["decomposer_agent"]
    assert agent["decomposer_system_prompt_profile"] == "teacher"
    assert agent["subagent_types"] == [
        {
            "agent_type_id": SUBAGENT_TYPE_ID,
            "assistant_id": "qwen35_4b_non_thinking",
            "url": "http://127.0.0.1:2031",
            "description": SUBAGENT_DESCRIPTION,
        }
    ]
    # The assistant id must name a graph the LangGraph server actually serves.
    graphs = json.loads((REPO_ROOT / "gyms/tau2_gym/subagents/langgraph.json").read_text())["graphs"]
    assert agent["subagent_types"][0]["assistant_id"] in graphs


def test_manager_proxy_carries_the_experiment_sampling() -> None:
    experiment = get_experiment("qwen38_flash_teacher_non_thinking")
    command = run_module.remote_proxy_command(
        8144, response_tool_parser="qwen3_xml", extra_body=experiment.manager_proxy_extra_body
    )
    body = json.loads(command[command.index("--extra-body-json") + 1])
    assert body["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": False}
    # The Responses path ignores chat_template_kwargs; reasoning.effort is what switches thinking.
    assert body["reasoning"] == {"effort": "none"}
    assert body["temperature"] == 0.7
    thinking = get_experiment("qwen38_flash_teacher_thinking").manager_proxy_extra_body
    assert thinking["reasoning"] == {"effort": "xhigh"}
    low = get_experiment("qwen38_flash_teacher_thinking_low").manager_proxy_extra_body
    assert low["reasoning"] == {"effort": "low"}
    assert low["include_reasoning"] is True
    assert "--response-tool-parser" not in run_module.remote_proxy_command(8143)


def test_dry_run_plans_an_opd_round(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    checkpoint = tmp_path / "round_001"
    run_module.main(
        ["--experiment", "opd_rollout", "--manager-checkpoint", str(checkpoint),
         "--tasks-per-domain", "2", "--num-repeats", "4", "--dry"]
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["run_name"] == "opd_rollout-decomposer_train_v2-k2-n4"
    assert plan["manager"][plan["manager"].index("serve") + 1] == str(checkpoint)
    assert plan["dataset"].endswith(f"decomposer_train_v2-{plan['pool_sha256']}-k2.decomposer.jsonl")
    assert plan["subagent_backend"] == "llm_proxy"
    assert plan["subagent_model_id"] == "Qwen/Qwen3.5-4B-unlooped"


def test_subagent_model_follows_the_backend_unless_given() -> None:
    assert run_module.resolve_subagent_model_id("llm_proxy", None) == "Qwen/Qwen3.5-4B-unlooped"
    assert run_module.resolve_subagent_model_id("local_vllm", None) == "Qwen/Qwen3.5-4B"
    assert run_module.resolve_subagent_model_id("llm_proxy", "Qwen/Other") == "Qwen/Other"
    ports = run_module.PortLayout()
    env = run_module.base_environment(ports, subagent_backend="llm_proxy", subagent_model_id="Qwen/Other")
    assert env["TAU2_GYM_SUBAGENT_MODEL_ID"] == "Qwen/Other"
    assert list(json.loads(env["TAU2_GYM_MODEL_BASE_URLS_JSON"])) == ["Qwen/Other"]
    command = run_module.subagent_vllm_command(
        ports, model_id="Qwen/Qwen3.5-4B", max_model_len=1024, gpu_memory_utilization=0.5
    )
    assert command[command.index("serve") + 1] == "Qwen/Qwen3.5-4B"


def test_remote_experiments_refuse_a_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="remote manager"):
        run_module.resolve_checkpoint(get_experiment("qwen38_flash_teacher_thinking"), str(tmp_path))
    with pytest.raises(SystemExit, match="needs --manager-checkpoint"):
        run_module.resolve_checkpoint(get_experiment("opd_rollout"), None)


def test_checkpoint_fingerprint_changes_when_weights_are_rewritten(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"a")
    first = run_module.checkpoint_fingerprint(tmp_path)
    weights.write_bytes(b"ab")
    assert run_module.checkpoint_fingerprint(tmp_path) != first


def test_validate_result_requires_the_full_grid(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rows = [{"_ng_task_index": t, "_ng_rollout_index": r, "reward": float(t == 0)} for t in range(2) for r in range(2)]
    rollouts.write_text("\n".join(json.dumps(row) for row in rows))
    assert run_module.validate_result(rollouts, expected_tasks=2, num_repeats=2)["pass_rate"] == 0.5
    with pytest.raises(RuntimeError, match="Incomplete"):
        run_module.validate_result(rollouts, expected_tasks=3, num_repeats=2)


def test_run_name_records_pool_and_subsample() -> None:
    experiment = replace(get_experiment("qwen38_flash_teacher_non_thinking"))
    assert (
        run_module.run_name(experiment, pool="decomposer_eval_v1", tasks_per_domain=None, num_repeats=3, port_offset=0)
        == "qwen38_flash_teacher_non_thinking-decomposer_eval_v1-n3"
    )


def test_unlooped_teacher_sampling_for_manager_and_subagents() -> None:
    experiment = get_experiment("qwen38_flash_thinking_low_teacher_qwen35_4b_unlooped")
    assert experiment.pool == "decomposer_train_v2"
    assert experiment.prompt_profile == "teacher"
    assert experiment.upstream_replays_reasoning
    assert experiment.manager_proxy_extra_body == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "include_reasoning": True,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        "reasoning": {"effort": "low"},
    }
    env = run_module.subagent_environment(experiment)
    assert run_module.subagent_sampling_record(env) == {
        "temperature": 0.7,
        "top_p": 0.8,
        "extra_body": {
            "top_k": 20,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "max_completion_tokens": 8192,
    }
    assert run_module.upstream_model_ids(
        experiment, subagent_backend="llm_proxy", subagent_model_id="Qwen/Qwen3.5-4B-unlooped"
    ) == ["Qwen/Qwen3.8-Flash-Next-NVFP4", "Qwen/Qwen3.5-4B-unlooped"]
    assert run_module.upstream_model_ids(
        experiment, subagent_backend="local_vllm", subagent_model_id="Qwen/Qwen3.5-4B"
    ) == ["Qwen/Qwen3.8-Flash-Next-NVFP4"]


def test_existing_experiments_keep_their_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    low = get_experiment("qwen38_flash_teacher_thinking_low")
    assert low.manager_proxy_extra_body["presence_penalty"] == 1.5
    assert run_module.subagent_environment(low) == {}
    # A value left in the shell never reaches the graph of an experiment without one.
    monkeypatch.setenv("DECOMPOSER_SUBAGENT_SAMPLING_JSON", '{"temperature": 0.1}')
    env = run_module.base_environment(
        run_module.PortLayout(), subagent_backend="llm_proxy", subagent_model_id="Qwen/Qwen3.5-4B-unlooped"
    )
    env.update(run_module.subagent_environment(low))
    assert "DECOMPOSER_SUBAGENT_SAMPLING_JSON" not in env
    assert run_module.subagent_sampling_record(env)["presence_penalty"] == 1.5


def test_subagent_graph_sends_the_experiment_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "external" / "Gym"))
    from gyms.qwen_sampling import QWEN35_UNLOOPED_NON_THINKING, subagent_sampling_environment
    from gyms.tau2_gym.subagents import graph

    captured: dict = {}
    monkeypatch.setattr(graph, "ChatVLLM", lambda **kwargs: captured.update(kwargs) or object())
    monkeypatch.setattr(graph, "create_agent", lambda **kwargs: object())
    monkeypatch.setattr(graph, "SUBAGENT_MAX_COMPLETION_TOKENS", 8192)
    for name, value in subagent_sampling_environment(QWEN35_UNLOOPED_NON_THINKING).items():
        monkeypatch.setenv(name, value)
    graph.qwen35_4b_non_thinking()
    assert captured["temperature"] == 0.7 and captured["top_p"] == 0.8
    assert "presence_penalty" not in captured
    assert captured["max_completion_tokens"] == 8192
    assert captured["extra_body"] == {
        "top_k": 20,
        "include_reasoning": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert captured["preserve_reasoning"] is False


def test_invalid_replay_flag_is_rejected() -> None:
    with pytest.raises(ValueError, match="upstream_replays_reasoning"):
        replace(get_experiment("qwen38_flash_teacher_non_thinking"), upstream_replays_reasoning=True)
