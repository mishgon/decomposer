import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
from langchain_core.messages import AIMessage
import pytest

from gyms.widesearch import agents, dataset, run
from gyms.browsecomp import agents as browsecomp_agents
from evals.widesearch import run as evaluation


def test_prepare_exports_only_english_questions(tmp_path):
    rows = [{"instance_id": f"ws_en_{i:03}", "query": "Question", "language": "en",
             "evaluation": "private metadata", "answer": "secret"} for i in range(100)]
    rows.append({"instance_id": "ws_zh_001", "query": "Chinese", "language": "zh"})
    (tmp_path / "widesearch.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "tasks.jsonl"
    tasks = dataset.prepare(tmp_path, output)
    assert len(tasks) == 100
    assert all(set(task) == {"task_id", "question"} for task in tasks)
    assert "secret" not in output.read_text()


def test_researchers_share_upstream_prompt_tools_and_registry_models(monkeypatch):
    calls = []
    monkeypatch.setattr(agents, "create_model", lambda model: model)
    monkeypatch.setattr(agents, "create_agent", lambda model, **kwargs: calls.append((model, kwargs)))
    agents.researcher_4b_thinking()
    agents.researcher_27b_thinking()
    agents.researcher()
    assert calls[0][0] == "lmrouter/qwen_3_5_4b_thinking"
    assert calls[1][0] == "lmrouter/qwen_3_8_27b_medium_thinking"
    for _, kwargs in calls:
        assert [t.name for t in kwargs["tools"]] == ["search_global", "text_browser_view"]
        assert kwargs["system_prompt"].startswith(agents.PROMPT)
        assert "BrowseComp" not in kwargs["system_prompt"]
    assert calls[0][1]["system_prompt"] == calls[1][1]["system_prompt"] == agents.PROMPT
    assert agents.AGENT_SYSTEM_PROMPT in calls[2][1]["system_prompt"]
    assert agents.decomposer_sft_1_2_0 is browsecomp_agents.decomposer_sft_1_2_0


def test_episode_uses_widesearch_server_and_saves_raw_answer(monkeypatch, tmp_path):
    invoke = AsyncMock()
    monkeypatch.setattr(run.browsecomp, "episode", invoke)
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    args = SimpleNamespace(agent="researcher_4b_thinking", episode=True, retrieval_url=run.BACKEND)
    asyncio.run(run.main(args))
    kwargs = invoke.call_args.kwargs
    assert kwargs["server_config"].parent.name == "widesearch"
    assert "markdown" in kwargs["final_format"]
    answer = "|name|value|\n|---|---|\n|A|1|"
    result = asyncio.run(kwargs["evaluator"]({}, answer, tmp_path))
    assert result["status"] == "pending"
    assert (tmp_path / "answer.md").read_text() == answer


def query():
    return evaluation.WideSearchQuery("test", "Question", {
        "required": ["name", "value"], "unique_columns": ["name"],
        "eval_pipeline": {"name": {"metric": ["in_match"]},
                          "value": {"metric": ["exact_match"]}},
    }, pd.DataFrame({"name": ["A", "B"], "value": ["1", "2"]}), "en")


@pytest.mark.parametrize("answer,row_f1,item_f1,success", [
    ("|name|value|\n|---|---|\n|A|1|\n|B|2|", 1, 1, 1),
    ("|name|value|\n|---|---|\n|A|1|", 2/3, 2/3, 0),
    ("|name|value|\n|---|---|\n|A|wrong|\n|B|2|", .5, .75, 0),
    ("No answer", 0, 0, 0),
    ("|name|value|\n|---|---|\n|C|3|", 0, 0, 0),
])
def test_upstream_table_metrics(monkeypatch, tmp_path, answer, row_f1, item_f1, success):
    monkeypatch.setattr(evaluation, "create_model", lambda _: None)
    value = evaluation.score(query(), answer, tmp_path)
    assert value["status"] == "evaluated"
    assert value["score"] == success
    assert value["f1_by_row"] == pytest.approx(row_f1)
    assert value["f1_by_item"] == pytest.approx(item_f1)


def test_judge_failure_is_unscored_and_raw_response_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "create_model", lambda _: SimpleNamespace(invoke=lambda _: AIMessage(content="bad JSON")))
    q = query()
    q.evaluation["eval_pipeline"]["name"]["metric"] = ["exact_match"]
    value = evaluation.score(q, "|name|value|\n|---|---|\n|A|1|", tmp_path)
    assert value["status"] == "evaluation_error"
    assert value["score"] is None
    assert json.loads((tmp_path / "judge_calls.json").read_text())[0]["response"]["content"] == "bad JSON"


def test_summary_keeps_missing_and_failed_attempts_in_denominator():
    manifest = {"tasks": ["a", "b"], "repetitions": 3, "episodes": [
        {"task": "a", "repetition": 1, "status": "finished"},
        {"task": "b", "repetition": 1, "status": "error"}]}
    results = {key: {"status": "evaluated", "score": 1, "f1_by_row": 1, "f1_by_item": 1}
               for key in ["a-r001", "b-r001"]}
    summary = evaluation.summarize(manifest, results)
    assert summary["metrics"]["score"] == {"avg": 1/6, "max": .5}


@pytest.mark.parametrize("scores", [{}, {"idx_0": 1}, {"idx_0": 1, "idx_1": .5},
                                    {"idx_0": 1, "idx_1": "1"}, {"idx_0": 1, "idx_1": True}])
def test_incomplete_or_nonbinary_judge_scores_are_unscored(monkeypatch, tmp_path, scores):
    monkeypatch.setattr(evaluation, "create_model", lambda _: SimpleNamespace(
        invoke=lambda _: AIMessage(content="```json\n" + json.dumps(scores) + "\n```")))
    q = query()
    q.evaluation["eval_pipeline"]["value"]["metric"] = ["llm_judge"]
    value = evaluation.score(q, "|name|value|\n|---|---|\n|A|1|\n|B|2|", tmp_path)
    assert value["status"] == "evaluation_error"
    assert value["score"] is None


def test_gold_changes_invalidate_cached_scores(monkeypatch, tmp_path):
    source = tmp_path / "source"
    (source / "widesearch_gold").mkdir(parents=True)
    (source / "widesearch.jsonl").write_text("metadata")
    gold = source / "widesearch_gold/test.csv"
    gold.write_text("name,value\nA,1\n")
    root = tmp_path / "run"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"tasks": ["test"], "repetitions": 1, "episodes": []}))
    monkeypatch.setattr(evaluation, "load_queries", lambda _: {"test": query()})
    args = SimpleNamespace(run=root, source=source)
    evaluation.main(args)
    gold.write_text("name,value\nA,2\n")
    with pytest.raises(ValueError, match="provenance changed"):
        evaluation.main(args)
