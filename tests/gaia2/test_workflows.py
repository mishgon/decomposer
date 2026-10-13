from __future__ import annotations

import json
from pathlib import Path

import pytest

from gyms.gaia2 import agents as langgraph_server
from gyms.gaia2 import run
from gyms.gaia2.prepare import (
    DATASET_REVISION,
    DOMAIN,
    SCENARIO_COUNT,
    SPLIT,
    validate_materialized_dataset,
    validate_materialized_filesystem,
    write_filesystem_manifest,
    write_materialized_dataset,
)
from gyms.gaia2.run import validate_result


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


def test_langgraph_server_forces_in_memory_runtime(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_run_server(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("langgraph_api.cli.run_server", fake_run_server)

    assert (
        langgraph_server.main(
            [
                "--port",
                "2028",
                "--n-jobs-per-worker",
                "16",
            ]
        )
        == 0
    )
    assert captured["graphs"] == {
        "decomposer": "gyms.gaia2.agents:decomposer",
        "qwen_3_5_4b_thinking": "gyms.gaia2.agents:qwen_3_5_4b_thinking",
    }
    assert captured["disable_persistence"] is True
    assert captured["reload"] is False
    assert captured["open_browser"] is False
    assert captured["allow_blocking"] is True


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


def test_result_validation_checks_task_coverage_and_logical_runs(tmp_path) -> None:
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


def test_single_repeat_results_have_no_run_numbers(tmp_path) -> None:
    # ARE omits run_number when it runs each scenario once (--num-repeats 1).
    output = tmp_path / "output"
    output.mkdir()
    rows = [
        {"task_id": scenario_id, "score": 1.0, "metadata": {"status": "success"}}
        for scenario_id in ("a", "b")
    ]
    (output / "output.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    metrics = validate_result(
        output, num_repeats=1, limit=None, scenario_count=2, scenario_ids=("a", "b")
    )
    assert metrics["passed_rollouts"] == 2
    # With several repeats a missing run number is still an error.
    with pytest.raises(ValueError, match="rollout records"):
        validate_result(
            output, num_repeats=3, limit=None, scenario_count=2, scenario_ids=("a", "b")
        )


def test_runtime_budget_and_episode_deadline(tmp_path):
    args = run.build_parser().parse_args(
        [
            "--episode-timeout",
            "120",
            "--agent-run-budget-seconds",
            "10",
            "--agent-shutdown-grace-seconds",
            "20",
            "--port-offset",
            "100",
            "--decomposer-base-url",
            "http://manager/v1",
            "--agent-base-url",
            "http://agent/v1",
        ]
    )
    command = run.are_command(
        args,
        benchmark=Path("are"),
        dataset=tmp_path,
        directory=tmp_path,
        judge_endpoint="http://judge/v1",
    )
    assert command[command.index("--scenario_timeout") + 1] == "120"
    assert command[command.index("--trace_dump_format") + 1] == "both"


def test_dry_run_does_not_start_services(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        run, "validate_preparation", lambda **kw: pytest.fail("dry must not prepare")
    )
    directory = tmp_path / "output"
    args = run.build_parser().parse_args(
        ["--dry", "--limit", "1", "--output-dir", str(directory)]
    )
    assert run.execute(args) == directory
    assert not directory.exists()
    assert json.loads(capsys.readouterr().out)["scenario_ids"] == ["scenario_000"]


def test_existing_results_are_not_overwritten(tmp_path):
    (tmp_path / "output.jsonl").write_text("existing")
    args = run.build_parser().parse_args(["--output-dir", str(tmp_path)])
    with pytest.raises(FileExistsError):
        run.execute(args)
    assert (tmp_path / "output.jsonl").read_text() == "existing"


@pytest.mark.parametrize("agent", ["decomposer", "qwen_3_5_4b_thinking"])
@pytest.mark.parametrize("interrupted", [False, True])
def test_execution_collects_results_and_cleans_up(
    tmp_path, monkeypatch, agent, interrupted
):
    task = "scenario_000"
    source = tmp_path / "source" / "execution"
    source.mkdir(parents=True)
    (source / f"{task}.json").write_text("{}")
    directory = tmp_path / "output"
    args = run.build_parser().parse_args(
        [
            "--agent",
            agent,
            "--limit",
            "1",
            "--num-repeats",
            "1",
            "--output-dir",
            str(directory),
        ]
    )
    monkeypatch.setenv("LLM_PROXY_URL", "http://judge/v1")
    monkeypatch.setenv("LLM_PROXY_MASTER_KEY", "judge-secret")
    monkeypatch.setattr(
        run,
        "validate_preparation",
        lambda **kw: {
            "gaia2": {
                "source_repo": str(tmp_path),
                "runtime": {"are_benchmark": "are"},
            },
        },
    )
    monkeypatch.setattr(run, "dataset_root", lambda *a: source.parent)
    monkeypatch.setattr(run, "wait_http", lambda *a: None)
    started = []
    stopped = []

    class Process:
        def wait(self):
            if interrupted:
                raise KeyboardInterrupt()
            (directory / "output.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": task,
                        "score": 0.0,
                        "metadata": {"has_exception": True},
                    }
                )
                + "\n"
            )
            return 0

    class Supervisor:
        def __init__(self, logs, env):
            assert env["OPENAI_API_KEY"] == "judge-secret"
            assert env["GAIA2_DECOMPOSER_MODEL_ID"] == args.decomposer_model_id
            assert env["GAIA2_AGENT_SERVER_URL"] == "http://127.0.0.1:8123"
            assert env["GAIA2_AGENT_RUN_BUDGET_SECONDS"] == "60"
            assert env["GAIA2_AGENT_SHUTDOWN_GRACE_SECONDS"] == "60"
            assert env["GAIA2_AGENT_MODEL_ID"] == args.agent_model_id
            assert env["GAIA2_AGENT_BASE_URL"] == (args.agent_base_url or "")

        def start(self, name, command, **kw):
            started.append((name, command, kw))
            return Process()

        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(run, "Supervisor", Supervisor)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            run.execute(args)
    else:
        assert run.execute(args) == directory
    assert stopped == [True]
    status = json.loads((directory / "run_status.json").read_text())
    assert status["state"] == ("failed" if interrupted else "complete")
    assert (directory / ".eval_done.json").exists() is not interrupted
    assert (directory / "inputs" / "execution" / f"{task}.json").is_file()
    command = started[-1][1]
    assert command[command.index("--agent") + 1] == "agent_server"
    assert "judge-secret" not in json.dumps(status)
    assert [name for name, _, _ in started] == ["agent_server", "are_benchmark"]
    plugin = json.loads((directory / "configuration" / "are_plugin.json").read_text())
    assert plugin["assistant_id"] == agent
    assert not (directory / "configuration" / "agents.json").exists()


@pytest.fixture(autouse=True)
def dataset_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps(
            {"scenarios": [{"scenario_id": f"scenario_{i:03d}"} for i in range(3)]}
        )
    )
    monkeypatch.setattr(run, "dataset_manifest", lambda domain: manifest)


def test_task_selection_validates_domain():
    args = run.build_parser().parse_args(
        ["--task", "scenario_002", "--task", "scenario_000"]
    )
    assert run.selected_scenario_ids(args) == ("scenario_000", "scenario_002")
    args.task = ["missing"]
    with pytest.raises(ValueError, match="Unknown tasks"):
        run.selected_scenario_ids(args)


def test_only_execution_and_search_are_supported():
    with pytest.raises(SystemExit):
        run.build_parser().parse_args(["--domain", "ambiguity"])
