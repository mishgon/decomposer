from pathlib import Path

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
