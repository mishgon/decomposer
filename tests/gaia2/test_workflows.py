from __future__ import annotations

import json
from argparse import Namespace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from gyms.gaia2 import partition as partition_module
from gyms.gaia2 import prepare
from gyms.gaia2 import langgraph_server
from gyms.gaia2.dataset import (
    validate_materialized_dataset,
    validate_materialized_filesystem,
    write_filesystem_manifest,
    write_materialized_dataset,
)
from gyms.gaia2.experiments import (
    AMBIGUITY_DOMAIN,
    BASE_IMAGE,
    DATASET_REVISION,
    DEEPSEEK_GEMMA_EXPERIMENT,
    DEEPSEEK_QWEN_AMBIGUITY_POLICY_EXPERIMENT,
    DEEPSEEK_QWEN_EXPERIMENT,
    DECOMPOSER_EXPERIMENT,
    DOMAIN,
    GEMMA4_E4B_TEXT_DEFAULTS_SIMPLE_EXPERIMENT,
    GEMMA4_TEXT_DEFAULTS_DECOMPOSER_EXPERIMENT,
    INSTANCE_TYPES_BY_NUM_GPUS,
    QWEN35_BASE_DECOMPOSER_EXPERIMENT,
    QWEN35_BASE_TEACHER_DECOMPOSER_EXPERIMENT,
    QWEN35_FINAL_MIXED_SFT_EXPERIMENT,
    QWEN35_FILTERED_SFT_EXPERIMENT,
    QWEN35_GAIA2_EXECUTION_ONLY_SFT_EXPERIMENT,
    QWEN35_GAIA2_SFT_EXPERIMENT,
    QWEN35_MIXED_SFT_EXPERIMENT,
    QWEN35_SFT_EXPERIMENT,
    QWEN35_TOOLATHLON_ONLY_SFT_EXPERIMENT,
    QWEN36_QWEN_EXPERIMENT,
    QWEN36_TEXT_DEFAULTS_DECOMPOSER_EXPERIMENT,
    QWEN35_4B_TEXT_DEFAULTS_SIMPLE_EXPERIMENT,
    SEARCH_DOMAIN,
    SCENARIO_COUNT,
    SIMPLE_DEEPSEEK_EXPERIMENT,
    SIMPLE_EXPERIMENT,
    SIMPLE_QWEN_2B_EXPERIMENT,
    SIMPLE_QWEN_9B_EXPERIMENT,
    SIMPLE_QWEN_EXPERIMENT,
    SPLIT,
    SPLIT_MANIFEST_NAME,
    collect_experiments,
    filesystem_dir,
    output_dir,
    partition_dataset_root,
    trace_output_dir,
)
from gyms.gaia2.partition import (
    SPLIT_MANIFEST_SHA256,
    load_split_manifest,
    materialize_partition_views,
    partition_scenarios,
    validate_partition_view,
    validate_split_against_source,
)
from gyms.gaia2.run import (
    DEFAULT_PORT_LAYOUT,
    Gaia2PortLayout,
    _base_environment,
    _dry_plan,
    _runtime_configs,
    aggregate_trace_manifest,
    archive_attempt,
    are_command,
    decomposer_vllm_commands,
    langgraph_command,
    langgraph_runtime_paths,
    openrouter_proxy_command,
    prompt_sha256,
    remote_manager_proxy_command,
    run_identity,
    selected_output_dir,
    selected_cuda_devices,
    simple_vllm_command,
    simple_sampling_parameters,
    select_prompt_profile,
    subagent_environment,
    execute_trace_generation,
    validate_trace_round,
    validate_result,
    validate_run_identity,
)
from gyms.gaia2.run_eval import build_payload, normalize_job_desc, redact_payload
from gyms.gaia2.snapshot_trace_prefix import create_trace_prefix_snapshot


def test_port_layout_offsets_each_gaia2_service_role() -> None:
    ports = Gaia2PortLayout(12000)

    assert ports.as_dict(SIMPLE_QWEN_EXPERIMENT) == {
        "offset": 12000,
        "simple_agent": 20100,
        "manager": None,
        "worker": None,
        "langgraph": None,
        "decomposer_service": None,
    }
    assert ports.as_dict(QWEN35_SFT_EXPERIMENT) == {
        "offset": 12000,
        "simple_agent": None,
        "manager": 20026,
        "worker": 20025,
        "langgraph": 14026,
        "decomposer_service": 20126,
    }
    assert ports.as_dict(QWEN36_QWEN_EXPERIMENT)["manager"] == 20142
    assert ports.as_dict(DEEPSEEK_QWEN_EXPERIMENT)["manager"] is None

    with pytest.raises(ValueError, match="non-negative"):
        Gaia2PortLayout(-1)
    with pytest.raises(ValueError, match="TCP port range"):
        Gaia2PortLayout(60000).as_dict(QWEN36_QWEN_EXPERIMENT)


def test_port_offset_threads_through_commands_and_runtime_configs(tmp_path) -> None:
    ports = Gaia2PortLayout(12000)
    simple_command = simple_vllm_command(SIMPLE_QWEN_EXPERIMENT, ports)
    assert simple_command[simple_command.index("--port") + 1] == "20100"
    openrouter_command = openrouter_proxy_command(SIMPLE_DEEPSEEK_EXPERIMENT, ports)
    assert openrouter_command[openrouter_command.index("--port") + 1] == "20140"

    manager, worker = decomposer_vllm_commands(QWEN35_SFT_EXPERIMENT, ports)
    assert manager is not None
    assert manager[manager.index("--port") + 1] == "20026"
    assert worker[worker.index("--port") + 1] == "20025"

    proxy = remote_manager_proxy_command(QWEN36_QWEN_EXPERIMENT, ports)
    assert proxy[proxy.index("--port") + 1] == "20142"
    langgraph, _ = langgraph_command(QWEN36_QWEN_EXPERIMENT, tmp_path, ports)
    assert langgraph[langgraph.index("--port") + 1] == "14034"
    assert subagent_environment(QWEN36_QWEN_EXPERIMENT, ports)[
        "GAIA2_SUBAGENT_ENDPOINT"
    ] == "http://127.0.0.1:20031/v1"

    service_path, plugin_path = _runtime_configs(
        Path(__file__).resolve().parents[2],
        tmp_path,
        QWEN36_QWEN_EXPERIMENT,
        ports,
    )
    service = json.loads(service_path.read_text(encoding="utf-8"))
    plugin = json.loads(plugin_path.read_text(encoding="utf-8"))
    assert service["manager"]["base_url"] == "http://127.0.0.1:20142/v1"
    assert service["subagent_types"][0]["url"] == "http://127.0.0.1:14034"
    assert plugin["service_url"] == "http://127.0.0.1:20134"

    are = are_command(
        SIMPLE_QWEN_EXPERIMENT,
        benchmark=tmp_path / "are-benchmark",
        dataset_root=tmp_path / "dataset",
        output=tmp_path / "output",
        judge_endpoint="https://judge.test/v1",
        num_repeats=3,
        limit=None,
        plugin_config=None,
        ports=ports,
    )
    assert are[are.index("--endpoint") + 1] == "http://127.0.0.1:20100/v1"


def test_port_offset_isolates_default_output_and_resume_identity(tmp_path) -> None:
    args = Namespace(
        purpose="evaluation",
        partition="test",
        num_repeats=3,
        rollout_offset=0,
        limit=2,
        output_dir=None,
    )
    directory = selected_output_dir(
        SIMPLE_QWEN_EXPERIMENT,
        args,
        domain=DOMAIN,
        prompt_profile=None,
        ports=Gaia2PortLayout(12000),
    )
    assert directory.parent.name.endswith("-port-offset-12000")
    assert directory.name == "smoke_2"

    args.purpose = "trace-generation"
    args.partition = "train"
    args.rollout_offset = 3
    trace_directory = selected_output_dir(
        DEEPSEEK_QWEN_EXPERIMENT,
        args,
        domain=DOMAIN,
        prompt_profile=None,
        ports=Gaia2PortLayout(24000),
    )
    assert trace_directory.parent.name.endswith("-port-offset-24000")
    assert trace_directory.name == "smoke_2"

    explicit = tmp_path / "explicit"
    args.output_dir = explicit
    assert selected_output_dir(
        DEEPSEEK_QWEN_EXPERIMENT,
        args,
        domain=DOMAIN,
        prompt_profile=None,
        ports=Gaia2PortLayout(24000),
    ) == explicit.resolve()

    legacy_identity = run_identity(
        SIMPLE_QWEN_EXPERIMENT,
        domain=DOMAIN,
        purpose="evaluation",
        partition="test",
        num_repeats=3,
        concurrency=4,
        limit=None,
    )
    legacy_identity.pop("port_offset")
    legacy_identity.pop("port_layout")
    marker = tmp_path / ".eval_done.json"
    marker.write_text(
        json.dumps({"state": "complete", **legacy_identity}) + "\n",
        encoding="utf-8",
    )
    expected_default = run_identity(
        SIMPLE_QWEN_EXPERIMENT,
        domain=DOMAIN,
        purpose="evaluation",
        partition="test",
        num_repeats=3,
        concurrency=4,
        limit=None,
        ports=DEFAULT_PORT_LAYOUT,
    )
    validate_run_identity(marker, expected_default, require_complete=True)
    with pytest.raises(ValueError, match="output identity mismatch"):
        validate_run_identity(
            marker,
            run_identity(
                SIMPLE_QWEN_EXPERIMENT,
                domain=DOMAIN,
                purpose="evaluation",
                partition="test",
                num_repeats=3,
                concurrency=4,
                limit=None,
                ports=Gaia2PortLayout(12000),
            ),
            require_complete=True,
        )


def test_port_offset_dry_plans_are_disjoint(tmp_path) -> None:
    cases = (
        (SIMPLE_QWEN_EXPERIMENT, Gaia2PortLayout(0), ("0",)),
        (QWEN35_SFT_EXPERIMENT, Gaia2PortLayout(12000), ("1", "2")),
        (QWEN36_QWEN_EXPERIMENT, Gaia2PortLayout(24000), ("3",)),
    )
    occupied: set[int] = set()
    for index, (experiment, ports, devices) in enumerate(cases):
        plan = _dry_plan(
            Path(__file__).resolve().parents[2],
            experiment,
            tmp_path / f"run-{index}",
            devices,
            3,
            None,
            ports=ports,
        )
        assert plan["port_offset"] == ports.offset
        active_ports = {
            value
            for key, value in plan["port_layout"].items()
            if key != "offset" and value is not None
        }
        assert occupied.isdisjoint(active_ports)
        occupied.update(active_ports)


def _rows():
    for index in range(SCENARIO_COUNT):
        scenario_id = f"scenario_{index:03d}"
        yield {
            "scenario_id": scenario_id,
            "split": SPLIT,
            "data": json.dumps(
                {"metadata": {"definition": {"scenario_id": scenario_id}}}
            ),
        }


def test_dataset_materialization_is_revisioned_and_integrity_checked(tmp_path) -> None:
    revision_root = tmp_path / DATASET_REVISION
    manifest = write_materialized_dataset(
        _rows(), revision_root, gaia2_revision="gaia-commit"
    )

    assert manifest["rows"] == SCENARIO_COUNT
    assert manifest["domain"] == DOMAIN
    assert manifest["split"] == SPLIT
    scenario = revision_root / SPLIT / DOMAIN / "scenario_000.json"
    assert scenario.is_file()
    assert (
        validate_materialized_dataset(revision_root)["aggregate_sha256"]
        == manifest["aggregate_sha256"]
    )

    scenario.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed or is missing"):
        validate_materialized_dataset(revision_root)


def test_execution_split_is_pinned_disjoint_and_isolates_complete_universes() -> None:
    manifest = load_split_manifest()
    train = partition_scenarios(manifest, "train")
    test = partition_scenarios(manifest, "test")
    full = partition_scenarios(manifest, "full")

    assert len(train) == 110
    assert len(test) == 50
    assert len(full) == 160
    assert {item["scenario_id"] for item in train}.isdisjoint(
        item["scenario_id"] for item in test
    )
    assert {item["universe"] for item in test} == {25, 26, 28}
    assert {item["universe"] for item in train}.isdisjoint({25, 26, 28})
    assert manifest["seed"] == 42
    assert manifest["dataset"]["revision"] == DATASET_REVISION
    assert partition_module.sha256_file(partition_module.SPLIT_MANIFEST_PATH) == (
        SPLIT_MANIFEST_SHA256
    )

    source = {
        "aggregate_sha256": manifest["dataset"]["aggregate_sha256"],
        "scenarios": [
            {
                "scenario_id": item["scenario_id"],
                "size": item["size"],
                "sha256": item["sha256"],
            }
            for item in full
        ],
    }
    validate_split_against_source(manifest, source)
    source["scenarios"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source scenarios"):
        validate_split_against_source(manifest, source)


def test_search_split_is_pinned_disjoint_and_domain_isolated() -> None:
    manifest = load_split_manifest(domain="search")
    train = partition_scenarios(manifest, "train")
    test = partition_scenarios(manifest, "test")

    assert manifest["name"] == "search-118-42-v1"
    assert manifest["dataset"]["domain"] == "search"
    assert manifest["dataset"]["aggregate_sha256"] == (
        SEARCH_DOMAIN.dataset_aggregate_sha256
    )
    assert len(train) == 118
    assert len(test) == 42
    assert {item["scenario_id"] for item in train}.isdisjoint(
        item["scenario_id"] for item in test
    )
    assert {item["universe"] for item in test} == {25, 26, 28}
    assert partition_dataset_root("test", "search") != partition_dataset_root(
        "test", "execution"
    )
    assert output_dir(
        SIMPLE_QWEN_EXPERIMENT,
        3,
        partition="test",
        domain="search",
    ).parts[-5:-1] == (
        "search",
        "partitions",
        "search-118-42-v1",
        "test",
    )


def test_ambiguity_split_is_pinned_disjoint_and_domain_isolated() -> None:
    manifest = load_split_manifest(domain="ambiguity")
    train = partition_scenarios(manifest, "train")
    test = partition_scenarios(manifest, "test")
    full = partition_scenarios(manifest, "full")

    assert manifest["name"] == "ambiguity-128-32-v1"
    assert manifest["dataset"]["domain"] == "ambiguity"
    assert manifest["dataset"]["aggregate_sha256"] == (
        AMBIGUITY_DOMAIN.dataset_aggregate_sha256
    )
    assert len(train) == 128
    assert len(test) == 32
    assert len(full) == 160
    assert {item["scenario_id"] for item in train}.isdisjoint(
        item["scenario_id"] for item in test
    )
    assert {item["universe"] for item in test} == {25, 26, 28}
    assert {item["universe"] for item in train}.isdisjoint({25, 26, 28})
    assert partition_dataset_root("test", "ambiguity") != partition_dataset_root(
        "test", "execution"
    )
    assert output_dir(
        SIMPLE_QWEN_EXPERIMENT,
        3,
        domain="ambiguity",
    ).parts[-2:] == ("ambiguity", "qwen35-4b-non-thinking-n3")


def test_partition_views_are_copies_and_checksum_validated(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    source_directory = source_root / SPLIT / DOMAIN
    source_directory.mkdir(parents=True)
    scenarios = []
    for universe, suffix, partition in (
        (21, "alpha", "train"),
        (22, "beta", "train"),
        (25, "gamma", "test"),
        (28, "delta", "test"),
    ):
        scenario_id = f"scenario_universe_{universe}_{suffix}"
        content = f"content-{scenario_id}".encode()
        (source_directory / f"{scenario_id}.json").write_bytes(content)
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "universe": universe,
                "partition": partition,
                "size": len(content),
                "sha256": sha256(content).hexdigest(),
            }
        )
    source_manifest = {
        "aggregate_sha256": "synthetic-source",
        "scenarios": [
            {
                "scenario_id": item["scenario_id"],
                "size": item["size"],
                "sha256": item["sha256"],
            }
            for item in scenarios
        ],
    }
    split_manifest = {
        "dataset": {"aggregate_sha256": "synthetic-source"},
        "scenarios": scenarios,
    }
    monkeypatch.setattr(
        partition_module,
        "validate_materialized_dataset",
        lambda _root, **_kwargs: source_manifest,
    )
    monkeypatch.setattr(
        partition_module,
        "load_split_manifest",
        lambda **_kwargs: split_manifest,
    )

    root = tmp_path / "views"
    views = materialize_partition_views(root=root, source_root=source_root)

    assert views["train"]["rows"] == 2
    assert views["test"]["rows"] == 2
    source_file = source_directory / f"{scenarios[0]['scenario_id']}.json"
    view_file = root / "train" / DOMAIN / source_file.name
    assert view_file.read_bytes() == source_file.read_bytes()
    assert not view_file.samefile(source_file)

    view_file.write_text("changed", encoding="utf-8")
    assert source_file.read_bytes().startswith(b"content-")
    with pytest.raises(ValueError, match="partition scenario changed"):
        validate_partition_view("train", root=root, source_root=source_root)


def test_filesystem_mirror_manifest_detects_asset_changes(tmp_path) -> None:
    directory = tmp_path / "demo_filesystem" / "Documents"
    directory.mkdir(parents=True)
    asset = directory / "example.txt"
    asset.write_text("original", encoding="utf-8")

    manifest = write_filesystem_manifest(tmp_path)

    assert manifest["file_count"] == 1
    assert manifest["total_bytes"] == len("original")
    assert (
        validate_materialized_filesystem(tmp_path)["aggregate_sha256"]
        == manifest["aggregate_sha256"]
    )

    asset.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum changed"):
        validate_materialized_filesystem(tmp_path)


def test_experiment_registry_contains_local_and_openrouter_profiles() -> None:
    assert collect_experiments() == [
        DECOMPOSER_EXPERIMENT,
        DEEPSEEK_GEMMA_EXPERIMENT,
        QWEN35_SFT_EXPERIMENT,
        QWEN35_MIXED_SFT_EXPERIMENT,
        QWEN35_FINAL_MIXED_SFT_EXPERIMENT,
        QWEN35_FILTERED_SFT_EXPERIMENT,
        QWEN35_GAIA2_SFT_EXPERIMENT,
        QWEN35_TOOLATHLON_ONLY_SFT_EXPERIMENT,
        QWEN35_GAIA2_EXECUTION_ONLY_SFT_EXPERIMENT,
        QWEN35_BASE_DECOMPOSER_EXPERIMENT,
        QWEN35_BASE_TEACHER_DECOMPOSER_EXPERIMENT,
        DEEPSEEK_QWEN_EXPERIMENT,
        DEEPSEEK_QWEN_AMBIGUITY_POLICY_EXPERIMENT,
        QWEN36_QWEN_EXPERIMENT,
        GEMMA4_TEXT_DEFAULTS_DECOMPOSER_EXPERIMENT,
        QWEN36_TEXT_DEFAULTS_DECOMPOSER_EXPERIMENT,
        SIMPLE_EXPERIMENT,
        SIMPLE_QWEN_EXPERIMENT,
        SIMPLE_QWEN_2B_EXPERIMENT,
        SIMPLE_QWEN_9B_EXPERIMENT,
        GEMMA4_E4B_TEXT_DEFAULTS_SIMPLE_EXPERIMENT,
        QWEN35_4B_TEXT_DEFAULTS_SIMPLE_EXPERIMENT,
        SIMPLE_DEEPSEEK_EXPERIMENT,
    ]
    assert BASE_IMAGE.endswith("py3.12-torch2.7.0:0.0.42")
    assert INSTANCE_TYPES_BY_NUM_GPUS == {
        1: "a100plus.1gpu.80vG.12C.182G",
        2: "a100plus.2gpu.80vG.24C.364G",
    }
    assert all(
        not experiment.manager_parallel_tool_calls
        for experiment in (
            DECOMPOSER_EXPERIMENT,
            DEEPSEEK_GEMMA_EXPERIMENT,
            QWEN35_SFT_EXPERIMENT,
            QWEN35_MIXED_SFT_EXPERIMENT,
            QWEN35_FINAL_MIXED_SFT_EXPERIMENT,
            QWEN35_FILTERED_SFT_EXPERIMENT,
            QWEN35_GAIA2_SFT_EXPERIMENT,
            QWEN35_BASE_DECOMPOSER_EXPERIMENT,
            QWEN35_BASE_TEACHER_DECOMPOSER_EXPERIMENT,
            DEEPSEEK_QWEN_EXPERIMENT,
            DEEPSEEK_QWEN_AMBIGUITY_POLICY_EXPERIMENT,
            QWEN36_QWEN_EXPERIMENT,
            GEMMA4_TEXT_DEFAULTS_DECOMPOSER_EXPERIMENT,
            QWEN36_TEXT_DEFAULTS_DECOMPOSER_EXPERIMENT,
        )
    )


def test_qwen36_teacher_uses_internal_proxy_and_existing_worker(tmp_path) -> None:
    experiment = QWEN36_QWEN_EXPERIMENT
    repo_root = Path(__file__).resolve().parents[2]
    assert experiment.manager_backend == "llm_proxy"
    assert experiment.manager_reasoning_mode == "service_default"
    assert experiment.manager_served_name == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert experiment.prompt_profile == "teacher"
    assert experiment.worker_checkpoint == DEEPSEEK_QWEN_EXPERIMENT.worker_checkpoint
    assert experiment.worker_served_name == "Qwen/Qwen3.5-4B"
    assert experiment.num_gpus == 1
    assert experiment.concurrency == 16

    command = remote_manager_proxy_command(experiment)
    assert "gyms.remote_model_proxy" in command
    assert command[command.index("--upstream-url-env") + 1] == "LLM_PROXY_URL"
    assert command[command.index("--api-key-env") + 1] == "LLM_PROXY_MASTER_KEY"
    assert command[command.index("--response-tool-parser") + 1] == "qwen3_xml"
    assert "--no-verify-tls" in command

    service_path, _ = _runtime_configs(repo_root, tmp_path / "result", experiment)
    service = json.loads(service_path.read_text())
    manager = service["manager"]
    assert manager["base_url"] == "http://127.0.0.1:8142/v1"
    assert manager["api_key"] == "EMPTY"
    assert manager["use_responses_api"] is True
    assert manager["parallel_tool_calls"] is False

    plan = _dry_plan(
        repo_root,
        experiment,
        tmp_path / "result",
        ("0",),
        3,
        None,
        purpose="evaluation",
        partition="test",
        concurrency=16,
    )
    assert plan["concurrency"] == 16
    assert "gyms.remote_model_proxy" in plan["services"][0]
    assert plan["gpu_assignments"] == {"worker_vllm": "0"}
    assert output_dir(SIMPLE_EXPERIMENT, 3).parts[-3:] == (
        SPLIT,
        DOMAIN,
        "gemma4-e4b-it-thinking-n3",
    )
    assert output_dir(SIMPLE_QWEN_EXPERIMENT, 3).parts[-1] == (
        "qwen35-4b-non-thinking-n3"
    )
    assert output_dir(SIMPLE_QWEN_2B_EXPERIMENT, 3).parts[-1] == (
        "qwen35-2b-base-non-thinking-n3"
    )
    assert output_dir(SIMPLE_QWEN_9B_EXPERIMENT, 3).parts[-1] == (
        "qwen35-9b-base-non-thinking-n3"
    )
    assert output_dir(SIMPLE_DEEPSEEK_EXPERIMENT, 3).parts[-1] == (
        "deepseek-v4-flash-0731-n3"
    )


def test_qwen36_text_defaults_are_explicitly_non_thinking(tmp_path) -> None:
    experiment = QWEN36_TEXT_DEFAULTS_DECOMPOSER_EXPERIMENT
    expected_proxy_body = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "max_output_tokens": 32768,
        "include_reasoning": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    assert experiment.prompt_profile == "teacher"
    assert experiment.manager_reasoning_mode == "non_thinking"
    assert experiment.max_model_len == 131072
    assert experiment.max_completion_tokens == 32768
    assert experiment.manager_max_model_calls == 200
    assert experiment.subagent_max_model_calls == 200
    assert experiment.remote_manager_extra_body == expected_proxy_body

    proxy = remote_manager_proxy_command(experiment)
    assert (
        json.loads(proxy[proxy.index("--extra-body-json") + 1])
        == expected_proxy_body
    )
    service_path, _ = _runtime_configs(Path.cwd(), tmp_path, experiment)
    service = json.loads(service_path.read_text())
    assert service["manager"]["temperature"] == 0.7
    assert service["manager"]["top_p"] == 0.8
    assert "presence_penalty" not in service["manager"]
    assert service["manager"]["max_completion_tokens"] == 32768
    assert service["manager"]["extra_body"] == expected_proxy_body
    assert service["manager_max_model_calls"] == 200
    assert service["manager_recursion_limit"] == 1000
    assert service["subagent_recursion_limit"] == 1000

    manager, worker = decomposer_vllm_commands(experiment)
    assert manager is None
    assert "--language-model-only" in worker
    assert subagent_environment(experiment)["GAIA2_SUBAGENT_MAX_MODEL_CALLS"] == "200"


def test_gemma_text_defaults_use_pinned_thinking_models() -> None:
    experiment = GEMMA4_TEXT_DEFAULTS_DECOMPOSER_EXPERIMENT
    assert experiment.manager_checkpoint.name == (
        "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
    )
    assert experiment.manager_checkpoint.is_dir()
    assert experiment.worker_checkpoint.is_dir()
    assert experiment.max_model_len == 131072
    assert experiment.max_completion_tokens == 32768
    assert experiment.manager_max_model_calls == 200
    assert experiment.subagent_max_model_calls == 200
    assert (experiment.temperature, experiment.top_p, experiment.top_k) == (
        1.0,
        0.95,
        64,
    )

    manager, worker = decomposer_vllm_commands(experiment)
    assert manager is not None
    for command in (manager, worker):
        assert command[command.index("--max-model-len") + 1] == "131072"
        assert "--language-model-only" in command
        assert '{"enable_thinking":true}' in command
    environment = subagent_environment(experiment)
    assert environment["GAIA2_SUBAGENT_MAX_COMPLETION_TOKENS"] == "32768"
    assert environment["GAIA2_SUBAGENT_MAX_MODEL_CALLS"] == "200"


def test_text_default_simple_profiles_match_context_completion_and_calls() -> None:
    gemma = GEMMA4_E4B_TEXT_DEFAULTS_SIMPLE_EXPERIMENT
    qwen = QWEN35_4B_TEXT_DEFAULTS_SIMPLE_EXPERIMENT
    for experiment in (gemma, qwen):
        assert experiment.max_model_len == 131072
        assert experiment.max_completion_tokens == 32768
        assert experiment.max_model_calls == 200
        command = simple_vllm_command(experiment)
        assert "--language-model-only" in command
    assert simple_sampling_parameters(gemma) == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 32768,
    }
    assert simple_sampling_parameters(qwen) == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 32768,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }
    assert output_dir(
        SIMPLE_EXPERIMENT,
        3,
        partition="test",
    ).parts[-4:] == (
        "partitions",
        SPLIT_MANIFEST_NAME,
        "test",
        "gemma4-e4b-it-thinking-n3",
    )


def test_langgraph_runtime_is_private_and_disables_file_persistence(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    student_output = tmp_path / "student"
    teacher_output = tmp_path / "teacher"

    _runtime_configs(repo_root, student_output, QWEN35_BASE_DECOMPOSER_EXPERIMENT)
    _runtime_configs(
        repo_root,
        teacher_output,
        QWEN35_BASE_TEACHER_DECOMPOSER_EXPERIMENT,
    )

    student_config, student_cwd = langgraph_runtime_paths(student_output)
    teacher_config, teacher_cwd = langgraph_runtime_paths(teacher_output)
    student_command, command_cwd = langgraph_command(
        QWEN35_BASE_DECOMPOSER_EXPERIMENT,
        student_output,
    )

    assert student_config != teacher_config
    assert student_cwd != teacher_cwd
    assert command_cwd == student_cwd
    assert student_command[student_command.index("--config") + 1] == str(
        student_config
    )
    assert repo_root not in student_cwd.parents

    for config_path in (student_config, teacher_config):
        config = langgraph_server.load_runtime_config(config_path)
        assert config == {
            "dependencies": ["."],
            "disable_persistence": True,
            "graphs": {
                "gaia2_worker": "gyms.gaia2.subagents.graphs:gaia2_worker",
            },
            "python_version": "3.12",
        }

    student_config.write_text(
        json.dumps({"graphs": {"gaia2_worker": "module:graph"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="persistence must be disabled"):
        langgraph_server.load_runtime_config(student_config)

    plan = _dry_plan(
        repo_root,
        QWEN35_BASE_DECOMPOSER_EXPERIMENT,
        student_output,
        ("0", "1"),
        3,
        None,
        purpose="evaluation",
        partition="full",
        concurrency=4,
        domain="ambiguity",
    )
    assert plan["langgraph_runtime"] == {
        "config": str(student_config),
        "working_directory": str(student_cwd),
        "file_persistence": False,
    }


def test_langgraph_server_forces_in_memory_runtime(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        json.dumps(
            {
                "disable_persistence": True,
                "graphs": {"gaia2_worker": "module:graph"},
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_server(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("langgraph_api.cli.run_server", fake_run_server)

    assert (
        langgraph_server.main(
            [
                "--config",
                str(config_path),
                "--port",
                "2028",
                "--n-jobs-per-worker",
                "16",
            ]
        )
        == 0
    )
    assert captured["graphs"] == {"gaia2_worker": "module:graph"}
    assert captured["disable_persistence"] is True
    assert captured["reload"] is False
    assert captured["open_browser"] is False
    assert captured["allow_blocking"] is True


def test_vllm_commands_use_current_e4b_thinking_profiles() -> None:
    simple = simple_vllm_command(SIMPLE_EXPERIMENT)
    manager, worker = decomposer_vllm_commands(DECOMPOSER_EXPERIMENT)

    assert "gemma4" in simple
    assert '{"enable_thinking":true}' in simple
    assert '{"enable_thinking":false}' in manager
    assert '{"enable_thinking":true}' in worker
    assert str(DECOMPOSER_EXPERIMENT.manager_checkpoint) in manager
    assert str(DECOMPOSER_EXPERIMENT.worker_checkpoint) in worker


def test_simple_qwen_uses_qwen_runtime_and_sampling_profile() -> None:
    for experiment in (
        SIMPLE_QWEN_EXPERIMENT,
        SIMPLE_QWEN_2B_EXPERIMENT,
        SIMPLE_QWEN_9B_EXPERIMENT,
    ):
        command = simple_vllm_command(experiment)

        assert str(experiment.checkpoint) in command
        assert command[command.index("--max-model-len") + 1] == "131072"
        assert command[command.index("--tool-call-parser") + 1] == "qwen3_xml"
        assert "--reasoning-parser" not in command
        assert "--language-model-only" not in command
        assert "--trust-remote-code" in command
        assert command[command.index("--gdn-prefill-backend") + 1] == "triton"
        assert '{"enable_thinking":false}' in command
        assert simple_sampling_parameters(experiment) == {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 4096,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        }


def test_simple_deepseek_uses_local_credential_proxy_without_gpu() -> None:
    experiment = SIMPLE_DEEPSEEK_EXPERIMENT

    assert experiment.requires_openrouter
    assert experiment.num_gpus == 0
    assert selected_cuda_devices(experiment, None) == ()
    assert experiment.remote_extra_body == {"reasoning": {"effort": "high"}}
    assert simple_sampling_parameters(experiment) == {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 4096,
    }
    with pytest.raises(ValueError, match="do not start a local vLLM"):
        simple_vllm_command(experiment)

    command = openrouter_proxy_command(experiment)
    assert "gyms.gaia2.openrouter_proxy" in command
    assert "OPENROUTER_API_KEY_DECOMPOSER" in command
    assert not any("sk-or-" in item for item in command)
    assert '{"reasoning":{"effort":"high"}}' in command

    plan = _dry_plan(Path.cwd(), experiment, Path("/tmp/deepseek"), (), 3, None)
    assert plan["gpu_assignments"] == {}
    assert len(plan["services"]) == 1
    assert "openrouter_proxy" in plan["services"][0]


def test_openrouter_decomposer_starts_only_the_configured_worker() -> None:
    gemma_manager, gemma_worker = decomposer_vllm_commands(DEEPSEEK_GEMMA_EXPERIMENT)
    qwen_manager, qwen_worker = decomposer_vllm_commands(DEEPSEEK_QWEN_EXPERIMENT)

    assert gemma_manager is None
    assert qwen_manager is None
    assert "gemma4" in gemma_worker
    assert "--language-model-only" in gemma_worker
    assert "qwen3_xml" in qwen_worker
    assert "--reasoning-parser" not in qwen_worker
    assert "--language-model-only" not in qwen_worker
    assert "--trust-remote-code" in qwen_worker
    assert qwen_worker[qwen_worker.index("--gdn-prefill-backend") + 1] == "triton"


@pytest.mark.parametrize(
    "experiment",
    [
        QWEN35_SFT_EXPERIMENT,
        QWEN35_MIXED_SFT_EXPERIMENT,
        QWEN35_FINAL_MIXED_SFT_EXPERIMENT,
        QWEN35_FILTERED_SFT_EXPERIMENT,
        QWEN35_GAIA2_SFT_EXPERIMENT,
    ],
)
def test_qwen_sft_decomposer_uses_qwen_manager_and_worker_profiles(
    experiment,
) -> None:
    manager, worker = decomposer_vllm_commands(experiment)

    assert manager is not None
    assert str(experiment.manager_checkpoint) in manager
    assert str(experiment.worker_checkpoint) in worker
    for command in (manager, worker):
        assert "qwen3_xml" in command
        assert "--reasoning-parser" not in command
        assert '{"enable_thinking":false}' in command
        assert command[command.index("--gdn-prefill-backend") + 1] == "triton"
    assert "--language-model-only" in manager
    assert "--trust-remote-code" not in manager
    assert "--language-model-only" not in worker
    assert "--trust-remote-code" in worker


def test_untuned_qwen_decomposer_matches_sft_two_gpu_topology() -> None:
    experiment = QWEN35_BASE_DECOMPOSER_EXPERIMENT

    assert experiment.name == ("qwen35-4b-base-non-thinking-qwen35-4b-non-thinking")
    assert experiment.num_gpus == 2
    assert experiment.prompt_profile == "student"
    assert experiment.manager_parallel_tool_calls is False
    assert experiment.manager_checkpoint == experiment.worker_checkpoint
    assert experiment.manager_port != experiment.worker_port
    assert (
        experiment.temperature,
        experiment.top_p,
        experiment.top_k,
        experiment.min_p,
        experiment.presence_penalty,
        experiment.repetition_penalty,
    ) == (0.7, 0.8, 20, 0.0, 1.5, 1.0)

    manager, worker = decomposer_vllm_commands(experiment)
    assert manager is not None
    for command, port in (
        (manager, experiment.manager_port),
        (worker, experiment.worker_port),
    ):
        assert str(experiment.worker_checkpoint) in command
        assert command[command.index("--port") + 1] == str(port)
        assert command[command.index("--max-model-len") + 1] == "131072"
        assert command[command.index("--tool-call-parser") + 1] == "qwen3_xml"
        assert command[command.index("--gdn-prefill-backend") + 1] == "triton"
        assert "--reasoning-parser" not in command
        assert "--language-model-only" not in command
        assert "--trust-remote-code" in command
        assert '{"enable_thinking":false}' in command

    plan = _dry_plan(Path.cwd(), experiment, Path("/tmp/output"), ("0", "1"), 3, None)
    assert plan["gpu_assignments"] == {
        "manager_vllm": "0",
        "worker_vllm": "1",
    }
    vllm_services = [service for service in plan["services"] if "vllm serve" in service]
    assert len(vllm_services) == 2


def test_untuned_qwen_teacher_decomposer_only_changes_prompt_identity() -> None:
    experiment = QWEN35_BASE_TEACHER_DECOMPOSER_EXPERIMENT

    assert experiment.name == (
        "qwen35-4b-base-non-thinking-teacher-qwen35-4b-non-thinking"
    )
    assert experiment.prompt_profile == "teacher"
    assert experiment.num_gpus == 2
    assert (
        experiment.manager_checkpoint
        == QWEN35_BASE_DECOMPOSER_EXPERIMENT.manager_checkpoint
    )
    assert (
        experiment.worker_checkpoint
        == QWEN35_BASE_DECOMPOSER_EXPERIMENT.worker_checkpoint
    )
    assert (
        experiment.manager_served_name
        == QWEN35_BASE_DECOMPOSER_EXPERIMENT.manager_served_name
    )
    assert (
        experiment.worker_served_name
        == QWEN35_BASE_DECOMPOSER_EXPERIMENT.worker_served_name
    )
    assert experiment.manager_thinking is False
    assert experiment.worker_thinking is False
    assert experiment.manager_parallel_tool_calls is False

    plan = _dry_plan(Path.cwd(), experiment, Path("/tmp/output"), ("0", "1"), 3, None)
    assert plan["decomposer_system_prompt_profile"] == "teacher"
    assert plan["gpu_assignments"] == {
        "manager_vllm": "0",
        "worker_vllm": "1",
    }
    assert output_dir(experiment, 3).parts[-1] == (
        "qwen35-4b-base-non-thinking-teacher-qwen35-4b-non-thinking-n3"
    )


def test_qwen_worker_uses_official_non_thinking_sampling() -> None:
    experiment = DEEPSEEK_QWEN_EXPERIMENT
    assert (
        experiment.temperature,
        experiment.top_p,
        experiment.top_k,
        experiment.min_p,
        experiment.presence_penalty,
        experiment.repetition_penalty,
    ) == (0.7, 0.8, 20, 0.0, 1.5, 1.0)
    assert subagent_environment(experiment) == {
        "GAIA2_SUBAGENT_MODEL": "Qwen/Qwen3.5-4B",
        "GAIA2_SUBAGENT_ENDPOINT": "http://127.0.0.1:8031/v1",
        "GAIA2_SUBAGENT_API_KEY": "EMPTY",
        "GAIA2_SUBAGENT_TEMPERATURE": "0.7",
        "GAIA2_SUBAGENT_TOP_P": "0.8",
        "GAIA2_SUBAGENT_TOP_K": "20",
        "GAIA2_SUBAGENT_MAX_COMPLETION_TOKENS": "4096",
        "GAIA2_SUBAGENT_THINKING": "0",
        "GAIA2_SUBAGENT_MIN_P": "0.0",
        "GAIA2_SUBAGENT_PRESENCE_PENALTY": "1.5",
        "GAIA2_SUBAGENT_REPETITION_PENALTY": "1.0",
    }


@pytest.mark.parametrize(
    ("experiment", "served_name"),
    [
        (
            QWEN35_SFT_EXPERIMENT,
            "decomposer/qwen35-4b-sft-workplace-v1-3765-32k",
        ),
        (
            QWEN35_MIXED_SFT_EXPERIMENT,
            "decomposer/qwen35-4b-sft-mixed-v1-partial-3983f605-327-32k",
        ),
        (
            QWEN35_FINAL_MIXED_SFT_EXPERIMENT,
            "decomposer/qwen35-4b-sft-mixed-v1-final-493c24c4-404",
        ),
        (
            QWEN35_FILTERED_SFT_EXPERIMENT,
            (
                "decomposer/qwen35-4b-sft-mixed-v1-final-493c24c4-404-"
                "filtered-p1-s279"
            ),
        ),
        (
            QWEN35_GAIA2_SFT_EXPERIMENT,
            ("decomposer/qwen35-4b-sft-mixed-v2-493c24c4-" "gaia2-110-n3-filtered-p2"),
        ),
        (
            QWEN35_TOOLATHLON_ONLY_SFT_EXPERIMENT,
            (
                "decomposer/qwen35-4b-sft-toolathlon-only-v1-493c24c4-"
                "teacher-prompt-filtered-32k"
            ),
        ),
        (
            QWEN35_GAIA2_EXECUTION_ONLY_SFT_EXPERIMENT,
            (
                "decomposer/qwen35-4b-sft-gaia2-execution-only-v1-110-n10-"
                "teacher-prompt-r1-balanced-32k"
            ),
        ),
    ],
)
def test_qwen_sft_manager_uses_official_non_thinking_sampling(
    tmp_path,
    experiment,
    served_name: str,
) -> None:
    service_path, _ = _runtime_configs(
        Path(__file__).resolve().parents[2],
        tmp_path,
        experiment,
    )
    manager = json.loads(service_path.read_text(encoding="utf-8"))["manager"]

    assert manager == {
        "model": served_name,
        "base_url": "http://127.0.0.1:8026/v1",
        "api_key": "EMPTY",
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "max_completion_tokens": 4096,
        "use_responses_api": False,
        "parallel_tool_calls": False,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def test_gemma_worker_sampling_is_unchanged() -> None:
    assert (
        DECOMPOSER_EXPERIMENT.temperature,
        DECOMPOSER_EXPERIMENT.top_p,
        DECOMPOSER_EXPERIMENT.top_k,
    ) == (1.0, 0.95, 64)
    environment = subagent_environment(DECOMPOSER_EXPERIMENT)
    assert "GAIA2_SUBAGENT_MIN_P" not in environment
    assert "GAIA2_SUBAGENT_PRESENCE_PENALTY" not in environment
    assert "GAIA2_SUBAGENT_REPETITION_PENALTY" not in environment


def test_openrouter_runtime_uses_teacher_responses_api(tmp_path) -> None:
    _, plugin_path = _runtime_configs(
        Path(__file__).resolve().parents[2],
        tmp_path,
        DEEPSEEK_GEMMA_EXPERIMENT,
    )
    plugin = json.loads(plugin_path.read_text(encoding="utf-8"))
    service = plugin["service_configuration"]

    assert service["decomposer_system_prompt_profile"] == "teacher"
    assert service["manager"] == {
        "model": "deepseek/deepseek-v4-flash-0731",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY_DECOMPOSER",
        "temperature": 1.0,
        "top_p": 1.0,
        "use_responses_api": True,
        "reasoning": {"effort": "high"},
        "timeout": 3300,
        "max_retries": 2,
        "parallel_tool_calls": False,
    }
    assert "path" not in plugin["model_configuration"]["manager"]
    assert plugin["model_configuration"]["manager"]["parallel_tool_calls"] is False


def test_ambiguity_policy_ablation_has_distinct_prompt_and_artifact_identity(
    tmp_path,
) -> None:
    experiment = DEEPSEEK_QWEN_AMBIGUITY_POLICY_EXPERIMENT

    assert experiment.manager_prompt_addendum_profile == "gaia2-ambiguity"
    assert experiment.worker_checkpoint == DEEPSEEK_QWEN_EXPERIMENT.worker_checkpoint
    assert experiment.manager_served_name == DEEPSEEK_QWEN_EXPERIMENT.manager_served_name
    assert prompt_sha256(experiment) != prompt_sha256(DEEPSEEK_QWEN_EXPERIMENT)
    assert output_dir(experiment, 3, domain="ambiguity") != output_dir(
        DEEPSEEK_QWEN_EXPERIMENT,
        3,
        domain="ambiguity",
    )

    service_path, _ = _runtime_configs(
        Path(__file__).resolve().parents[2],
        tmp_path,
        experiment,
    )
    service = json.loads(service_path.read_text(encoding="utf-8"))
    assert service["decomposer_system_prompt_profile"] == "teacher"
    assert (
        service["decomposer_system_prompt_addendum_profile"]
        == "gaia2-ambiguity"
    )

    identity = run_identity(
        experiment,
        domain="ambiguity",
        purpose="evaluation",
        partition="full",
        num_repeats=3,
        concurrency=4,
        limit=None,
    )
    assert identity["decomposer_system_prompt_addendum_profile"] == (
        "gaia2-ambiguity"
    )
    assert identity["decomposer_system_prompt_sha256"] == prompt_sha256(experiment)


def test_decomposer_dry_plan_records_sequential_manager_tool_calls(tmp_path) -> None:
    plan = _dry_plan(
        Path(__file__).resolve().parents[2],
        DEEPSEEK_QWEN_EXPERIMENT,
        tmp_path,
        ("0",),
        1,
        1,
    )

    assert plan["manager_parallel_tool_calls"] is False


def test_openrouter_preparation_hashes_only_the_local_worker(monkeypatch) -> None:
    calls = []

    def fake_validate_checkpoint(path, *, full_hashes):
        calls.append((path, full_hashes))
        return {"path": str(path)}

    monkeypatch.setattr(prepare, "validate_checkpoint", fake_validate_checkpoint)

    models = prepare.experiment_models(
        DEEPSEEK_QWEN_EXPERIMENT,
        full_hashes=False,
    )

    assert calls == [(DEEPSEEK_QWEN_EXPERIMENT.worker_checkpoint, False)]
    assert models["manager"] == {
        "backend": "openrouter",
        "model": "deepseek/deepseek-v4-flash-0731",
    }


@pytest.mark.parametrize(
    "experiment",
    [
        QWEN35_SFT_EXPERIMENT,
        QWEN35_MIXED_SFT_EXPERIMENT,
        QWEN35_FINAL_MIXED_SFT_EXPERIMENT,
        QWEN35_FILTERED_SFT_EXPERIMENT,
        QWEN35_GAIA2_SFT_EXPERIMENT,
        QWEN35_TOOLATHLON_ONLY_SFT_EXPERIMENT,
        QWEN35_GAIA2_EXECUTION_ONLY_SFT_EXPERIMENT,
    ],
)
def test_qwen_sft_preparation_hashes_manager_and_worker(
    monkeypatch,
    experiment,
) -> None:
    calls = []

    def fake_validate_checkpoint(path, *, full_hashes):
        calls.append((path, full_hashes))
        return {"path": str(path)}

    monkeypatch.setattr(prepare, "validate_checkpoint", fake_validate_checkpoint)

    models = prepare.experiment_models(
        experiment,
        full_hashes=False,
    )

    assert calls == [
        (experiment.worker_checkpoint, False),
        (experiment.manager_checkpoint, False),
    ]
    assert models == {
        "worker": {"path": str(experiment.worker_checkpoint)},
        "manager": {"path": str(experiment.manager_checkpoint)},
    }


def test_simple_qwen_preparation_hashes_policy_only(monkeypatch) -> None:
    calls = []

    def fake_validate_checkpoint(path, *, full_hashes):
        calls.append((path, full_hashes))
        return {"path": str(path)}

    monkeypatch.setattr(prepare, "validate_checkpoint", fake_validate_checkpoint)

    models = prepare.experiment_models(
        SIMPLE_QWEN_EXPERIMENT,
        full_hashes=False,
    )

    assert calls == [(SIMPLE_QWEN_EXPERIMENT.checkpoint, False)]
    assert models == {"policy": {"path": str(SIMPLE_QWEN_EXPERIMENT.checkpoint)}}


def test_simple_deepseek_preparation_records_remote_policy_without_hashing(
    monkeypatch,
) -> None:
    def unexpected_validate_checkpoint(path, *, full_hashes):
        raise AssertionError(f"Unexpected checkpoint validation: {path}")

    monkeypatch.setattr(prepare, "validate_checkpoint", unexpected_validate_checkpoint)

    assert prepare.experiment_models(
        SIMPLE_DEEPSEEK_EXPERIMENT,
        full_hashes=False,
    ) == {
        "policy": {
            "backend": "openrouter",
            "model": "deepseek/deepseek-v4-flash-0731",
        }
    }


def test_are_commands_use_local_execution_dataset(tmp_path) -> None:
    benchmark = tmp_path / "are-benchmark"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    simple = are_command(
        SIMPLE_EXPERIMENT,
        benchmark=benchmark,
        dataset_root=dataset,
        output=output,
        judge_endpoint="https://judge.test/v1",
        num_repeats=3,
        limit=2,
        plugin_config=None,
    )
    decomposer = are_command(
        DECOMPOSER_EXPERIMENT,
        benchmark=benchmark,
        dataset_root=dataset,
        output=output,
        judge_endpoint="https://judge.test/v1",
        num_repeats=3,
        limit=None,
        plugin_config=tmp_path / "plugin.json",
    )

    assert simple[simple.index("-d") + 1] == str(dataset)
    assert "--hf-dataset" not in simple
    assert simple[simple.index("--num_runs") + 1] == "3"
    assert simple[simple.index("--limit") + 1] == "2"
    assert "native_tools" in simple
    assert "gyms.gaia2.plugin:create_plugin" in decomposer
    assert "thread" in decomposer


def test_search_are_commands_support_simple_and_decomposer_agents(tmp_path) -> None:
    dataset = tmp_path / "search"
    simple = are_command(
        SIMPLE_QWEN_EXPERIMENT,
        benchmark=tmp_path / "are-benchmark",
        dataset_root=dataset,
        output=tmp_path / "simple",
        judge_endpoint="https://judge.test/v1",
        num_repeats=3,
        limit=None,
        plugin_config=None,
        domain="search",
    )
    decomposer = are_command(
        QWEN35_GAIA2_SFT_EXPERIMENT,
        benchmark=tmp_path / "are-benchmark",
        dataset_root=dataset,
        output=tmp_path / "decomposer",
        judge_endpoint="https://judge.test/v1",
        num_repeats=3,
        limit=None,
        plugin_config=tmp_path / "plugin.json",
        domain="search",
    )

    assert simple[simple.index("--config") + 1] == "search"
    assert decomposer[decomposer.index("--config") + 1] == "search"
    assert simple[simple.index("-d") + 1] == str(dataset)
    assert decomposer[decomposer.index("-d") + 1] == str(dataset)
    assert "native_tools" in simple
    assert "gyms.gaia2.plugin:create_plugin" in decomposer


def test_ambiguity_are_commands_support_simple_and_decomposer_agents(tmp_path) -> None:
    dataset = tmp_path / "ambiguity"
    simple = are_command(
        SIMPLE_QWEN_EXPERIMENT,
        benchmark=tmp_path / "are-benchmark",
        dataset_root=dataset,
        output=tmp_path / "simple",
        judge_endpoint="https://judge.test/v1",
        num_repeats=3,
        limit=None,
        plugin_config=None,
        domain="ambiguity",
    )
    decomposer = are_command(
        DEEPSEEK_QWEN_EXPERIMENT,
        benchmark=tmp_path / "are-benchmark",
        dataset_root=dataset,
        output=tmp_path / "decomposer",
        judge_endpoint="https://judge.test/v1",
        num_repeats=3,
        limit=None,
        plugin_config=tmp_path / "plugin.json",
        domain="ambiguity",
    )

    assert simple[simple.index("--config") + 1] == "ambiguity"
    assert decomposer[decomposer.index("--config") + 1] == "ambiguity"
    assert "native_tools" in simple
    assert "gyms.gaia2.plugin:create_plugin" in decomposer


def test_runtime_environment_separates_judge_and_local_vllm_credentials(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "inherited-key")

    environment = _base_environment(
        tmp_path / "decomposer",
        tmp_path / "gaia2",
        tmp_path / "output",
        "judge-key",
    )

    assert "VLLM_API_KEY" not in environment
    assert environment["OPENAI_API_KEY"] == "judge-key"
    assert environment["ARE_JUDGE_SAMPLING_PARAMS"] == '{"temperature":0.0}'
    assert environment["DEMO_FS_PATH"] == str(filesystem_dir())


def test_result_validation_keeps_failed_rollouts_in_fixed_denominator(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    rows = [
        {"task_id": "a", "score": 1.0, "metadata": {"has_exception": False}},
        {
            "task_id": "a",
            "score": 0.0,
            "metadata": {"has_exception": True, "exception_type": "RemoteError"},
        },
        {"task_id": "a", "score": 0.0, "metadata": {"has_exception": False}},
    ]
    (output / "output.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    metrics = validate_result(output, num_repeats=3, limit=1)

    assert metrics["rollout_rows"] == 3
    assert metrics["passed_rollouts"] == 1
    assert metrics["failed_rollouts"] == 2
    assert metrics["fixed_denominator_score"] == pytest.approx(1 / 3)
    assert metrics["exception_types"] == {"RemoteError": 1}


def test_result_validation_checks_partition_coverage_and_logical_runs(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    rows = [
        {
            "task_id": scenario_id,
            "score": float(run_number == 1),
            "metadata": {"run_number": run_number},
        }
        for scenario_id in ("a", "b")
        for run_number in (1, 2, 3)
    ]
    (output / "output.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows)),
        encoding="utf-8",
    )

    metrics = validate_result(
        output,
        num_repeats=3,
        limit=None,
        scenario_count=2,
        scenario_ids=("a", "b"),
    )
    assert metrics["rollout_rows"] == 6

    rows[-1]["metadata"]["run_number"] = 2
    (output / "output.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate GAIA2 rollout"):
        validate_result(
            output,
            num_repeats=3,
            limit=None,
            scenario_count=2,
            scenario_ids=("a", "b"),
        )


def test_mlspace_payload_uses_registry_gpu_type_and_redactable_judge_key(
    tmp_path,
) -> None:
    payload = build_payload(
        QWEN35_FILTERED_SFT_EXPERIMENT,
        tmp_path / "staged",
        num_repeats=3,
        limit=None,
        author="sukhorukov",
        base_image="image",
        priority="high",
        force=False,
        judge_environment={
            "LLM_PROXY_URL": "https://judge.test/v1",
            "LLM_PROXY_MASTER_KEY": "secret",
        },
        proxy_environment={},
        openrouter_key="<not-set>",
    )

    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[2]
    assert payload["priority_class"] == "high"
    assert "--num-repeats 3" in payload["script"]
    assert "--cuda-visible-devices 0,1" in payload["script"]
    assert payload["env_variables"]["LLM_PROXY_MASTER_KEY"] == "secret"
    assert normalize_job_desc(payload["job_desc"]) == normalize_job_desc(
        payload["job_desc"] + " @someone"
    )


def test_heldout_mlspace_payload_uses_pinned_test_partition(tmp_path) -> None:
    payload = build_payload(
        QWEN35_GAIA2_SFT_EXPERIMENT,
        tmp_path / "staged",
        num_repeats=3,
        limit=None,
        author="sukhorukov",
        base_image="image",
        priority="high",
        force=False,
        judge_environment={
            "LLM_PROXY_URL": "https://judge.test/v1",
            "LLM_PROXY_MASTER_KEY": "secret",
        },
        proxy_environment={},
        openrouter_key="<not-set>",
        purpose="evaluation",
        partition="test",
        concurrency=4,
    )

    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[2]
    assert payload["priority_class"] == "high"
    assert f"{SPLIT_MANIFEST_NAME}-test" in payload["job_desc"]
    assert "--purpose evaluation" in payload["script"]
    assert "--partition test" in payload["script"]
    assert "--num-repeats 3" in payload["script"]
    assert "--concurrency 4" in payload["script"]


def test_search_mlspace_payload_is_domain_and_holdout_specific(tmp_path) -> None:
    payload = build_payload(
        QWEN35_GAIA2_SFT_EXPERIMENT,
        tmp_path / "staged",
        num_repeats=3,
        limit=None,
        author="sukhorukov",
        base_image="image",
        priority="high",
        force=False,
        judge_environment={
            "LLM_PROXY_URL": "https://judge.test/v1",
            "LLM_PROXY_MASTER_KEY": "secret",
        },
        proxy_environment={},
        openrouter_key="<not-set>",
        purpose="evaluation",
        partition="test",
        concurrency=4,
        domain="search",
    )

    assert "gaia2-validation-search" in payload["job_desc"]
    assert "search-118-42-v1-test" in payload["job_desc"]
    assert "--domain search" in payload["script"]
    assert "--partition test" in payload["script"]


@pytest.mark.parametrize(
    "experiment",
    (
        SIMPLE_QWEN_EXPERIMENT,
        DEEPSEEK_QWEN_EXPERIMENT,
        DEEPSEEK_QWEN_AMBIGUITY_POLICY_EXPERIMENT,
    ),
)
def test_ambiguity_full_mlspace_payload_is_canonical_and_high_priority(
    tmp_path, experiment
) -> None:
    payload = build_payload(
        experiment,
        tmp_path / "staged",
        num_repeats=3,
        limit=None,
        author="sukhorukov",
        base_image="image",
        priority="high",
        force=False,
        judge_environment={
            "LLM_PROXY_URL": "https://judge.test/v1",
            "LLM_PROXY_MASTER_KEY": "secret",
        },
        proxy_environment={"HTTPS_PROXY": "https://proxy.test"},
        openrouter_key="openrouter-secret",
        purpose="evaluation",
        partition="full",
        concurrency=4,
        domain="ambiguity",
    )

    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[1]
    assert payload["priority_class"] == "high"
    assert "gaia2-validation-ambiguity" in payload["job_desc"]
    assert "--domain ambiguity" in payload["script"]
    assert "--partition full" in payload["script"]
    assert "--num-repeats 3" in payload["script"]
    assert "--concurrency 4" in payload["script"]
    assert "ARE_EXTRA_SYSTEM_PROMPT" not in payload["env_variables"]
    assert "ARE_EXTRA_SYSTEM_PROMPT_FILE" not in payload["env_variables"]


def test_completion_marker_identity_rejects_cross_domain_reuse(tmp_path) -> None:
    marker = tmp_path / ".eval_done.json"
    execution_identity = run_identity(
        SIMPLE_QWEN_EXPERIMENT,
        domain="execution",
        purpose="evaluation",
        partition="test",
        num_repeats=3,
        concurrency=4,
        limit=None,
    )
    marker.write_text(
        json.dumps({"state": "complete", **execution_identity}) + "\n",
        encoding="utf-8",
    )
    validate_run_identity(marker, execution_identity, require_complete=True)

    search_identity = run_identity(
        SIMPLE_QWEN_EXPERIMENT,
        domain="search",
        purpose="evaluation",
        partition="test",
        num_repeats=3,
        concurrency=4,
        limit=None,
    )
    with pytest.raises(ValueError, match="output identity mismatch"):
        validate_run_identity(marker, search_identity, require_complete=True)


@pytest.mark.parametrize("domain", ("search", "ambiguity"))
def test_non_execution_trace_generation_is_rejected_before_startup(domain) -> None:
    with pytest.raises(ValueError, match=f"{domain} does not support trace generation"):
        execute_trace_generation(
            Path.cwd(),
            Namespace(domain=domain),
        )


def test_openrouter_mlspace_payload_uses_one_gpu_and_redacts_credentials(
    tmp_path,
) -> None:
    payload = build_payload(
        DEEPSEEK_QWEN_EXPERIMENT,
        tmp_path / "staged",
        num_repeats=1,
        limit=1,
        author="sukhorukov",
        base_image="image",
        priority="high",
        force=False,
        judge_environment={
            "LLM_PROXY_URL": "https://judge.test/v1",
            "LLM_PROXY_MASTER_KEY": "judge-secret",
        },
        proxy_environment={"HTTPS_PROXY": "http://user:pass@proxy.test"},
        openrouter_key="openrouter-secret",
    )

    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[1]
    assert "--cuda-visible-devices 0" in payload["script"]
    assert payload["env_variables"]["OPENROUTER_API_KEY_DECOMPOSER"] == (
        "openrouter-secret"
    )
    redacted = redact_payload(payload)
    assert redacted["env_variables"]["OPENROUTER_API_KEY_DECOMPOSER"] == "<redacted>"
    assert redacted["env_variables"]["HTTPS_PROXY"] == "<redacted>"


def test_remote_simple_agent_is_local_only_for_mlspace_launcher(tmp_path) -> None:
    with pytest.raises(ValueError, match="local-only remote experiment"):
        build_payload(
            SIMPLE_DEEPSEEK_EXPERIMENT,
            tmp_path / "staged",
            num_repeats=3,
            limit=None,
            author="sukhorukov",
            base_image="image",
            priority="high",
            force=False,
            judge_environment={
                "LLM_PROXY_URL": "https://judge.test/v1",
                "LLM_PROXY_MASTER_KEY": "judge-secret",
            },
            proxy_environment={"HTTPS_PROXY": "http://proxy.test"},
            openrouter_key="openrouter-secret",
        )


def test_gaia_prompt_override_is_propagated_and_output_isolated(tmp_path) -> None:
    selected = select_prompt_profile(QWEN35_GAIA2_SFT_EXPERIMENT, "teacher")
    assert selected.prompt_profile == "teacher"
    assert output_dir(
        selected,
        3,
        partition="test",
        prompt_profile="teacher",
    ).name.endswith("-prompt-teacher")
    payload = build_payload(
        QWEN35_GAIA2_SFT_EXPERIMENT,
        tmp_path,
        num_repeats=3,
        limit=None,
        author="sukhorukov",
        base_image="image",
        priority="high",
        force=False,
        judge_environment={
            "LLM_PROXY_URL": "https://judge.test/v1",
            "LLM_PROXY_MASTER_KEY": "judge-secret",
        },
        proxy_environment={},
        openrouter_key="unused",
        partition="test",
        prompt_profile="teacher",
    )
    assert "--prompt-profile teacher" in payload["script"]
    assert "prompt-teacher" in payload["job_desc"]
    with pytest.raises(ValueError, match="only valid for Decomposer"):
        select_prompt_profile(SIMPLE_EXPERIMENT, "teacher")


def _write_trace_round(
    trace_directory: Path,
    logical_rollout_number: int,
    scenario_ids: tuple[str, ...],
    *,
    failed_scenarios: frozenset[str] = frozenset(),
    completed: bool = True,
    native_run_number: int | None = 1,
) -> Path:
    round_directory = trace_directory / f"round_{logical_rollout_number:02d}"
    round_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario_id in reversed(scenario_ids):
        trace_id = None
        if scenario_id not in failed_scenarios:
            native_suffix = native_run_number if native_run_number is not None else 0
            filename = f"{scenario_id}_run_{native_suffix}_deadbeef.json"
            for trace_format in ("hf", "lite"):
                path = round_directory / trace_format / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            sidecar = (
                round_directory
                / "decomposer_sidecars"
                / f"{scenario_id}__run{native_suffix}.json"
            )
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text("{}\n", encoding="utf-8")
            trace_id = str(round_directory / "hf" / filename)
        rows.append(
            {
                "task_id": scenario_id,
                "trace_id": trace_id,
                "score": 0.0 if scenario_id in failed_scenarios else 1.0,
                "metadata": {
                    "scenario_id": scenario_id,
                    "status": (
                        "failed" if scenario_id in failed_scenarios else "success"
                    ),
                    "has_exception": scenario_id in failed_scenarios,
                    "exception_type": (
                        "HTTPStatusError" if scenario_id in failed_scenarios else None
                    ),
                },
            }
        )
        if native_run_number is not None:
            rows[-1]["metadata"]["run_number"] = native_run_number
    (round_directory / "output.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    if completed:
        (round_directory / ".round_done.json").write_text(
            json.dumps({"logical_rollout_number": logical_rollout_number}) + "\n",
            encoding="utf-8",
        )
    return round_directory


def test_trace_rounds_use_logical_numbers_four_through_ten_and_aggregate_failures(
    tmp_path,
) -> None:
    scenario_ids = ("scenario_a", "scenario_b")
    logical_rollouts = tuple(range(4, 11))
    for logical_rollout in logical_rollouts:
        _write_trace_round(
            tmp_path,
            logical_rollout,
            scenario_ids,
            failed_scenarios=(
                frozenset({"scenario_b"}) if logical_rollout == 7 else frozenset()
            ),
        )

    metrics, records = aggregate_trace_manifest(
        tmp_path,
        logical_rollout_numbers=logical_rollouts,
        scenario_ids=scenario_ids,
    )

    assert metrics["attempted_rollouts"] == 14
    assert metrics["unique_attempted_rollouts"] == 14
    assert metrics["rollouts_per_scenario"] == 7
    assert [record["logical_rollout_number"] for record in records] == [
        logical_rollout for logical_rollout in logical_rollouts for _ in scenario_ids
    ]
    failed = next(
        record
        for record in records
        if record["logical_rollout_number"] == 7
        and record["scenario_id"] == "scenario_b"
    )
    assert failed["reward"] == 0.0
    assert failed["exception_type"] == "HTTPStatusError"
    assert failed["sidecar"] is None
    assert failed["hf_trace"] is None
    assert failed["lite_trace"] is None
    assert len((tmp_path / "trace_manifest.jsonl").read_text().splitlines()) == 14


def test_trace_prefix_snapshot_is_compact_complete_and_immutable(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "snapshot"
    scenario_ids = ("scenario_a", "scenario_b")
    for logical in (4, 5):
        _write_trace_round(
            source,
            logical,
            scenario_ids,
            failed_scenarios=(
                frozenset({"scenario_b"}) if logical == 5 else frozenset()
            ),
        )
    marker = create_trace_prefix_snapshot(
        source,
        output,
        logical_rollout_numbers=(4, 5),
        scenario_ids=scenario_ids,
    )
    assert marker["state"] == "complete"
    assert marker["attempted_rollouts"] == 4
    assert marker["passed_rollouts"] == 3
    rows = [
        json.loads(line)
        for line in (output / "trace_manifest.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4
    assert all("hf_trace" not in row and "lite_trace" not in row for row in rows)
    assert not (output / "round_04" / "hf").exists()
    passing = [row for row in rows if row["reward"] == 1.0]
    assert all((output / row["sidecar"]).is_file() for row in passing)
    with pytest.raises(FileExistsError, match="already exists"):
        create_trace_prefix_snapshot(
            source,
            output,
            logical_rollout_numbers=(4, 5),
            scenario_ids=scenario_ids,
        )


def test_trace_dry_plan_dispatches_round_robin_with_one_native_run(tmp_path) -> None:
    plan = _dry_plan(
        Path.cwd(),
        DEEPSEEK_QWEN_EXPERIMENT,
        tmp_path,
        ("0",),
        7,
        None,
        purpose="trace-generation",
        partition="train",
        concurrency=10,
        rollout_offset=3,
    )

    assert plan["logical_rollout_numbers"] == list(range(4, 11))
    assert plan["concurrency"] == 10
    assert len(plan["services"]) == 3
    assert len(plan["are_rounds"]) == 7
    for logical_rollout, round_plan in zip(range(4, 11), plan["are_rounds"]):
        assert round_plan["logical_rollout_number"] == logical_rollout
        assert "--num_runs 1" in round_plan["command"]
        assert "--max_concurrent_scenarios 10" in round_plan["command"]
        assert f"round_{logical_rollout:02d}" in round_plan["command"]


@pytest.mark.parametrize(
    "initial_round_state", ["marked", "complete_unmarked", "partial_unmarked"]
)
def test_trace_execution_resumes_completed_round_and_reuses_services(
    tmp_path, monkeypatch, initial_round_state
) -> None:
    trace_directory = tmp_path / "trace"
    ports = Gaia2PortLayout(12000)
    scenario_ids = ("scenario_a", "scenario_b")
    initial_round = _write_trace_round(
        trace_directory,
        4,
        scenario_ids,
        completed=initial_round_state == "marked",
        native_run_number=None,
    )
    (trace_directory / "run_status.json").write_text(
        json.dumps(
            {
                "state": "failed",
                **run_identity(
                    DEEPSEEK_QWEN_EXPERIMENT,
                    domain=DOMAIN,
                    purpose="trace-generation",
                    partition="train",
                    num_repeats=2,
                    concurrency=10,
                    limit=None,
                    rollout_offset=3,
                    ports=ports,
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if initial_round_state == "partial_unmarked":
        output = initial_round / "output.jsonl"
        output.write_text(output.read_text().splitlines()[0] + "\n")
    started_services: list[str] = []
    started_service_details: dict[str, tuple[list[str], Path]] = {}
    are_commands: list[list[str]] = []
    readiness_urls: list[str] = []

    class Process:
        def poll(self):
            return None

    class FakeSupervisor:
        def __init__(self, _logs, _env):
            pass

        def start(self, name, command, *, cwd, env=None):
            started_services.append(name)
            started_service_details[name] = (command, cwd)
            return Process()

        def assert_running(self):
            return None

        def stop(self):
            return None

    def fake_subprocess_run(command, **kwargs):
        if command[0] == "nvidia-smi":
            return SimpleNamespace(stdout="GPU metadata")
        are_commands.append(command)
        round_directory = Path(command[command.index("--output_dir") + 1])
        logical_rollout = int(round_directory.name.removeprefix("round_"))
        _write_trace_round(
            trace_directory,
            logical_rollout,
            scenario_ids,
            completed=False,
        )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(
        "gyms.gaia2.run.partition_scenario_ids",
        lambda _part, **_kwargs: scenario_ids,
    )
    monkeypatch.setattr(
        "gyms.gaia2.run.partition_dataset_root",
        lambda _part, *_args, **_kwargs: tmp_path / "dataset",
    )
    monkeypatch.setattr(
        "gyms.gaia2.run.validate_preparation",
        lambda _experiment, **_kwargs: {
            "gaia2": {
                "staged_repo": str(tmp_path / "gaia2"),
                "runtime": {"are_benchmark": str(tmp_path / "are-benchmark")},
            }
        },
    )
    monkeypatch.setattr("gyms.gaia2.run.check_judge", lambda *_args: None)
    monkeypatch.setattr(
        "gyms.gaia2.run.wait_http",
        lambda url, *_args: readiness_urls.append(url),
    )
    monkeypatch.setattr("gyms.gaia2.run.Supervisor", FakeSupervisor)
    monkeypatch.setattr("gyms.gaia2.run.subprocess.run", fake_subprocess_run)
    monkeypatch.setattr("gyms.gaia2.run.git", lambda *_args: "commit")
    monkeypatch.setenv("LLM_PROXY_URL", "https://judge.test/v1")
    monkeypatch.setenv("LLM_PROXY_MASTER_KEY", "judge-key")
    monkeypatch.setenv("OPENROUTER_API_KEY_DECOMPOSER", "openrouter-key")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.test")
    args = Namespace(
        experiment=DEEPSEEK_QWEN_EXPERIMENT.name,
        purpose="trace-generation",
        partition="train",
        num_repeats=2,
        rollout_offset=3,
        concurrency=10,
        limit=None,
        dry=False,
        force=False,
        output_dir=trace_directory,
        cuda_visible_devices=("0",),
        domain=DOMAIN,
        port_offset=12000,
    )

    assert execute_trace_generation(tmp_path, args) == 0

    assert started_services == [
        "worker_vllm",
        "langgraph_subagent",
        "decomposer_service",
    ]
    langgraph_config, langgraph_cwd = langgraph_runtime_paths(trace_directory)
    langgraph_command_argv, started_langgraph_cwd = started_service_details[
        "langgraph_subagent"
    ]
    assert started_langgraph_cwd == langgraph_cwd
    assert langgraph_command_argv[langgraph_command_argv.index("--config") + 1] == str(
        langgraph_config
    )
    assert langgraph_command_argv[langgraph_command_argv.index("--port") + 1] == (
        "14034"
    )
    worker_command = started_service_details["worker_vllm"][0]
    assert worker_command[worker_command.index("--port") + 1] == "20031"
    service_command_argv = started_service_details["decomposer_service"][0]
    assert service_command_argv[service_command_argv.index("--port") + 1] == "20134"
    assert readiness_urls == [
        "http://127.0.0.1:20031/v1/models",
        "http://127.0.0.1:14034/ok",
        "http://127.0.0.1:20134/health",
    ]
    assert langgraph_server.load_runtime_config(langgraph_config)[
        "disable_persistence"
    ] is True
    expected_are_commands = 2 if initial_round_state == "partial_unmarked" else 1
    assert len(are_commands) == expected_are_commands
    assert all(
        command[command.index("--num_runs") + 1] == "1" for command in are_commands
    )
    assert (
        are_commands[0][are_commands[0].index("--max_concurrent_scenarios") + 1] == "10"
    )
    assert (trace_directory / "round_04" / ".round_done.json").is_file()
    assert (trace_directory / "round_05" / ".round_done.json").is_file()
    assert (trace_directory / ".trace_done.json").is_file()
    completion = json.loads((trace_directory / ".trace_done.json").read_text())
    assert completion["schema_version"] == 2
    assert completion["port_offset"] == 12000
    assert completion["port_layout"] == ports.as_dict(DEEPSEEK_QWEN_EXPERIMENT)
    assert len((trace_directory / "trace_manifest.jsonl").read_text().splitlines()) == 4
    round_marker = json.loads(
        (trace_directory / "round_04" / ".round_done.json").read_text()
    )
    assert bool(round_marker.get("recovered_from_unmarked_artifacts")) is (
        initial_round_state == "complete_unmarked"
    )
    if initial_round_state == "partial_unmarked":
        assert Path(round_marker["archived_attempt"]).is_dir()


def test_full_run_archive_preserves_nested_smoke_output(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke_1"
    smoke.mkdir(parents=True)
    (smoke / ".eval_done.json").write_text("{}")
    (tmp_path / "run_status.json").write_text("{}")

    archive = archive_attempt(tmp_path)

    assert archive is not None
    assert (archive / "run_status.json").is_file()
    assert (smoke / ".eval_done.json").is_file()


def test_trace_mlspace_payload_is_exactly_one_high_priority_gpu(tmp_path) -> None:
    payload = build_payload(
        DEEPSEEK_QWEN_EXPERIMENT,
        tmp_path / "staged",
        num_repeats=7,
        limit=None,
        author="sukhorukov",
        base_image="image",
        priority="high",
        force=False,
        judge_environment={
            "LLM_PROXY_URL": "https://judge.test/v1",
            "LLM_PROXY_MASTER_KEY": "judge-secret",
        },
        proxy_environment={"HTTPS_PROXY": "http://proxy.test"},
        openrouter_key="openrouter-secret",
        purpose="trace-generation",
        partition="train",
        concurrency=10,
        rollout_offset=3,
    )

    assert payload["instance_type"] == "a100plus.1gpu.80vG.12C.182G"
    assert payload["priority_class"] == "high"
    assert payload["n_workers"] == 1
    assert payload["processes_per_worker"] == 1
    assert "--purpose trace-generation" in payload["script"]
    assert "--partition train" in payload["script"]
    assert "--num-repeats 7" in payload["script"]
    assert "--concurrency 10" in payload["script"]
    assert "--rollout-offset 3" in payload["script"]
    assert "--cuda-visible-devices 0" in payload["script"]
