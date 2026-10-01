from pathlib import Path
import json

from evals.wideseek.run import summarize
from gyms.wideseek.run import create_parser


def test_harness_cli_and_legacy_mode():
    parser = create_parser()
    assert parser.parse_args(["--harness", "react", "--output", "/tmp/raw"]).harness == "react"
    assert parser.parse_args(["--mode", "simple", "--output", "/tmp/raw"]).mode == "simple"


def test_summary_keeps_judge_errors_separate_and_excludes_judge_tokens():
    def row(score, status):
        return {"evaluation": {"score": score, "metric": "item_f1"},
                "status": status, "started_at": 10, "agent_finished_at": 20,
                "usage": {"researcher": {"input_tokens": 5, "output_tokens": 2},
                          "judge": {"input_tokens": 100, "output_tokens": 50}}}
    result = summarize([row(0.5, "finished"), row(None, "error")])
    assert result["scored"] == result["unscored"] == 1
    assert result["mean_native_score"] == 0.5
    assert result["mean_native_score_infra_zero"] == 0.25
    assert result["agent_tokens"] == {"input_tokens": 10, "output_tokens": 4}
    assert result["normal_finishes"] == 1
    assert summarize([])["mean_native_score"] is None


def test_gym_has_no_evaluation_workflow_dependency():
    source = Path("gyms/wideseek/run.py").read_text()
    assert "from evals" not in source
    assert "import evals" not in source
    assert "from sft" not in source


def test_collection_indexes_only_finished_scored_traces(tmp_path):
    from sft.wideseek.run import successful_traces
    for task, status, score, cleanup, has_trace in (
        ("good", "finished", .95, [], True),
        ("partial", "finished", .8, [], True),
        ("unscored", "finished", None, [], True),
        ("stopped", "timeout", 1., [], True),
        ("cleanup", "finished", 1., ["still active"], True),
        ("missing", "finished", 1., [], False),
    ):
        attempt = tmp_path / "decomposer" / task / "attempt-001"
        execution = attempt / "execution-test"
        execution.mkdir(parents=True)
        (attempt / "result.json").write_text(json.dumps({
            "task_id": task, "attempt": 1, "status": status, "evaluation": {"score": score},
            "cleanup_errors": cleanup, "execution_directory": execution.name}))
        if has_trace:
            (execution / "trace.json").write_text('{}')
    rows = successful_traces(tmp_path, "decomposer", .9)
    assert len(rows) == 1
    assert rows[0]["task_id"] == "good"
    assert rows[0]["trace"] == "decomposer/good/attempt-001/execution-test/trace.json"


def test_models_use_shared_registry_without_sampling_environment(monkeypatch):
    from decomposer.models import create_model
    from gyms.wideseek.runtime import DEFAULT_SUBAGENT, model, model_metadata
    assert model is create_model
    monkeypatch.setenv("WS_MODEL", "retired/model")
    monkeypatch.setenv("LLM_PROXY_MASTER_KEY", "test-secret")
    policy = model(DEFAULT_SUBAGENT)
    metadata = model_metadata(policy)
    assert metadata["model_name"] == "Qwen/Qwen3.5-4B-unlooped"
    assert metadata["temperature"] == .6
    assert metadata["top_p"] == .95
    assert metadata["extra_body"]["chat_template_kwargs"]["enable_thinking"]
    assert metadata["preserve_reasoning"]
    assert "test-secret" not in json.dumps(metadata)
    import asyncio
    from gyms.wideseek.runtime import close_model
    asyncio.run(close_model(policy))
