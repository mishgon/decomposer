from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from data.sft.adapters.gaia2 import read_gaia2_source
from data.sft.builder import load_build_spec
from data.sft.schema import SelectionSpec, SourceSpec
from decomposer.core import build_decomposer_chat_tools
from decomposer.prompts import DECOMPOSER_SYSTEM_PROMPT


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def _split_manifest(path: Path) -> Path:
    scenarios = [
        {
            "scenario_id": f"scenario_universe_{universe}_{suffix}",
            "universe": universe,
            "partition": partition,
            "sha256": str(universe) * 64,
            "size": 1,
        }
        for universe, partition in ((21, "train"), (25, "test"))
        for suffix in ("a", "b")
    ]
    _write_json(
        path,
        {
            "name": "fixture-2-2-v1",
            "dataset": {"revision": "gaia-data-revision"},
            "partitions": {
                "full": {"rows": 4, "universes": [21, 25]},
                "train": {"rows": 2, "universes": [21]},
                "test": {"rows": 2, "universes": [25]},
            },
            "scenarios": scenarios,
        },
    )
    return path


def _manager_messages() -> list[dict]:
    return [
        {"type": "human", "data": {"content": "Complete the scenario."}},
        {
            "type": "ai",
            "data": {
                "content": [
                    {
                        "type": "reasoning",
                        "content": [{"type": "reasoning_text", "text": "Delegate."}],
                    },
                    {"type": "text", "text": "I will delegate."},
                ],
                "tool_calls": [
                    {
                        "id": "spawn-a",
                        "name": "spawn_subagent",
                        "args": {
                            "subagent_type_id": "gaia2_worker",
                            "prompt": "Do A.",
                        },
                    },
                    {
                        "id": "spawn-b",
                        "name": "spawn_subagent",
                        "args": {
                            "subagent_type_id": "gaia2_worker",
                            "prompt": "Do B.",
                        },
                    },
                ],
                "invalid_tool_calls": [],
            },
        },
        {
            "type": "tool",
            "data": {
                "content": "run b",
                "tool_call_id": "spawn-b",
                "name": "spawn_subagent",
            },
        },
        {
            "type": "tool",
            "data": {
                "content": "run a",
                "tool_call_id": "spawn-a",
                "name": "spawn_subagent",
            },
        },
        {
            "type": "ai",
            "data": {
                "content": [],
                "tool_calls": [{"id": "wait", "name": "wait", "args": {}}],
                "invalid_tool_calls": [],
            },
        },
        {
            "type": "tool",
            "data": {"content": "done", "tool_call_id": "wait", "name": "wait"},
        },
        {
            "type": "ai",
            "data": {
                "content": [{"type": "text", "text": "Finished."}],
                "tool_calls": [],
                "invalid_tool_calls": [],
            },
        },
    ]


def _sidecar(scenario_id: str, native_run_number: int | None) -> dict:
    return {
        "scenario_id": scenario_id,
        "run_number": native_run_number,
        "configuration": {
            "decomposer_system_prompt_profile": "teacher",
            "model_configuration": {
                "manager": {
                    "served_name": "deepseek/deepseek-v4-flash-0731",
                    "thinking": True,
                },
                "subagent": {"served_name": "Qwen/Qwen3.5-4B", "thinking": False},
            },
        },
        "decomposer_revision": "decomposer-revision",
        "gaia2_revision": "gaia2-revision",
        "turns": [
            {
                "manager": {
                    "trace": {
                        "manager_messages": _manager_messages(),
                        "runtime_context": {
                            "scenario_id": scenario_id,
                            "run_number": native_run_number,
                        },
                    }
                }
            }
        ],
    }


def _source(
    root: Path,
    split_manifest: Path,
    *,
    trace_format: str = "gaia2_evaluation_v1",
    logical_rollout_numbers: tuple[int, ...] = (1, 2),
    expected_native_rollouts: int = 8,
) -> SourceSpec:
    return SourceSpec(
        id="gaia2-fixture",
        adapter="gaia2",
        path=root,
        benchmark="gaia2",
        environment="gaia2_execution",
        partition="train",
        teacher="deepseek-v4-flash-0731",
        trace_format=trace_format,
        subagent_type_aliases={"gaia2_worker": "qwen35_4b_non_thinking"},
        expected_native_rollouts=expected_native_rollouts,
        expected_candidates=2 * len(logical_rollout_numbers),
        require_completed_run=True,
        selection={"policy": "exact_reward", "success_reward": 1.0},
        gaia2={
            "split_manifest": split_manifest,
            "expected_scenarios": 2,
            "logical_rollout_numbers": logical_rollout_numbers,
        },
    )


def _read(source: SourceSpec):
    tools = build_decomposer_chat_tools(
        [
            {
                "subagent_type_id": "qwen35_4b_non_thinking",
                "assistant_id": "qwen35_4b_non_thinking",
                "description": "Fixture worker.",
            }
        ]
    )
    return read_gaia2_source(
        source,
        SelectionSpec(policy="all_rewards", invalid_policy="exclude"),
        system_prompt=DECOMPOSER_SYSTEM_PROMPT,
        canonical_tools=tools,
        canonical_subagent_type_ids=frozenset({"qwen35_4b_non_thinking"}),
    )


def test_gaia2_evaluation_filters_binary_reward_and_holdout(tmp_path: Path) -> None:
    split = _split_manifest(tmp_path / "split.json")
    source_dir = tmp_path / "source"
    rows: list[dict] = []
    for universe in (21, 25):
        for suffix in ("a", "b"):
            scenario_id = f"scenario_universe_{universe}_{suffix}"
            for run_number in (1, 2):
                reward = float(universe == 25 or (suffix == "a" and run_number == 1))
                rows.append(
                    {
                        "task_id": scenario_id,
                        "score": reward,
                        "metadata": {
                            "scenario_id": scenario_id,
                            "run_number": run_number,
                            "status": "success" if reward else "failed",
                            "has_exception": False,
                        },
                    }
                )
                if universe == 21 and reward:
                    _write_json(
                        source_dir
                        / "decomposer_sidecars"
                        / f"{scenario_id}__run{run_number}.json",
                        _sidecar(scenario_id, run_number),
                    )
    _write_jsonl(source_dir / "output.jsonl", rows)
    _write_json(
        source_dir / ".eval_done.json",
        {
            "state": "complete",
            "kind": "decomposer",
            "decomposer_system_prompt_profile": "teacher",
            "num_repeats": 2,
            "decomposer_commit": "decomposer-revision",
            "gaia2_commit": "gaia2-revision",
            "metrics": {"rollout_rows": 8},
        },
    )

    result = _read(_source(source_dir, split))

    assert result.counts["rollouts"] == 4
    assert result.counts["eligible"] == 1
    assert result.counts["excluded_reward"] == 3
    assert len(result.records) == 1
    record = result.records[0]
    assert record.group_id == "gaia2:scenario_universe_21_a"
    assert record.outcome.reward == 1.0
    assert record.messages[0]["content"] == DECOMPOSER_SYSTEM_PROMPT
    spawn_messages = [
        message
        for message in record.messages
        if message["role"] == "assistant"
        and message.get("tool_calls")
        and message["tool_calls"][0]["function"]["name"] == "spawn_subagent"
    ]
    assert len(spawn_messages) == 2
    assert spawn_messages[0]["teacher_reasoning"] == "Delegate."
    assert all(
        message["tool_calls"][0]["function"]["arguments"]["subagent_type_id"]
        == "qwen35_4b_non_thinking"
        for message in spawn_messages
    )
    assert result.source_manifest["layout"]["holdout_rollouts"] == 4
    assert result.source_manifest["binary_reward_counts"] == {"0": 3, "1": 1}


def test_gaia2_trace_manifest_requires_terminal_full_grid(tmp_path: Path) -> None:
    split = _split_manifest(tmp_path / "split.json")
    source_dir = tmp_path / "source"
    rows = []
    for suffix in ("a", "b"):
        scenario_id = f"scenario_universe_21_{suffix}"
        for logical in (4, 5):
            sidecar = (
                f"round_{logical:02d}/decomposer_sidecars/{scenario_id}__run0.json"
            )
            reward = 0.0 if suffix == "b" and logical == 5 else 1.0
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "logical_rollout_number": logical,
                    "native_run_number": 0,
                    "reward": reward,
                    "status": "success" if reward else "failed",
                    "has_exception": False,
                    "exception_type": None,
                    "sidecar": sidecar if reward else None,
                }
            )
            if reward:
                _write_json(source_dir / sidecar, _sidecar(scenario_id, None))
    _write_jsonl(source_dir / "trace_manifest.jsonl", rows)
    source = _source(
        source_dir,
        split,
        trace_format="gaia2_trace_manifest_v1",
        logical_rollout_numbers=(4, 5),
        expected_native_rollouts=4,
    )

    with pytest.raises(ValueError, match="not terminal"):
        _read(source)

    _write_json(
        source_dir / ".trace_done.json",
        {
            "state": "complete",
            "decomposer_commit": "decomposer-revision",
            "gaia2_commit": "gaia2-revision",
        },
    )
    result = _read(source)
    assert len(result.records) == 3
    assert result.counts["excluded_reward"] == 1
    assert {record.source.rollout_id for record in result.records} == {"r04", "r05"}
    assert {record.attributes["native_run_number"] for record in result.records} == {
        0
    }
    assert result.source_manifest["logical_rollout_numbers"] == [4, 5]

    tampered = deepcopy(rows)
    tampered[0]["reward"] = 0.95
    _write_jsonl(source_dir / "trace_manifest.jsonl", tampered)
    with pytest.raises(ValueError, match="binary numeric reward"):
        _read(source)


def test_gaia2_mixed_spec_and_split_pin_future_task_membership() -> None:
    spec = load_build_spec(
        "data/sft/specs/"
        "decomposer_mixed_deepseek_qwen35_4b_nonthinking_"
        "v2_gaia2_execution_110_n3_filtered_32k.yaml"
    ).spec
    assert spec.split.strategy == "pinned"
    assert spec.split.manifest is not None
    assert spec.tokenization is not None
    assert spec.tokenization.revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    gaia = spec.sources[2]
    assert gaia.adapter == "gaia2"
    assert gaia.expected_native_rollouts == 480
    assert gaia.expected_candidates == 330
    assert gaia.selection is not None and gaia.selection.policy == "exact_reward"
    assert gaia.gaia2 is not None
    assert gaia.gaia2.logical_rollout_numbers == (1, 2, 3)

    manifest = json.loads(spec.split.manifest.read_text(encoding="utf-8"))
    assert manifest["summary"] == {
        "groups": 1361,
        "train_groups": 1225,
        "validation_groups": 136,
        "groups_by_source": {
            "gaia2-execution-deepseek-v4-flash-0731-qwen35-4b-nonthinking-n3": 110,
            "toolathlon-deepseek-v4-flash-0731-qwen35-4b-nonthinking-n1": 298,
            "workplace-deepseek-v4-flash-0731-qwen35-4b-nonthinking-n3": 953,
        },
    }
    gaia_groups = [
        group for group in manifest["groups"] if group["category"] == "gaia2_execution"
    ]
    assert len(gaia_groups) == 110
    assert sum(group["partition"] == "validation" for group in gaia_groups) == 11
