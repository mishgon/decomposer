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


def test_researchers_share_browsecomp_prompt_tools_and_registry_models(monkeypatch):
    calls = []
    monkeypatch.setattr(browsecomp_agents, "create_model", lambda model: model)
    monkeypatch.setattr(browsecomp_agents, "create_agent", lambda model, **kwargs: calls.append((model, kwargs)))
    agents.researcher_4b_thinking()
    agents.researcher_27b_thinking()
    agents.researcher()
    assert calls[0][0] == "vllm/qwen_3_5_4b_thinking"
    assert calls[1][0] == "lmrouter/qwen_3_8_27b_medium_thinking"
    for _, kwargs in calls:
        assert [t.name for t in kwargs["tools"]] == ["search", "fetch"]
        assert "BrowseComp-Plus" in kwargs["system_prompt"]
    assert calls[0][1]["system_prompt"] == calls[1][1]["system_prompt"] == calls[2][1]["system_prompt"]
    assert browsecomp_agents.AGENT_SYSTEM_PROMPT in calls[2][1]["system_prompt"]
    assert agents.decomposer_sft_1_2_0 is browsecomp_agents.decomposer_sft_1_2_0
    assert agents.decomposer_sft_1_4_0 is browsecomp_agents.decomposer_sft_1_4_0
    assert "decomposer_sft_1_4_0" in run.AGENTS


def test_episode_uses_widesearch_server_and_saves_raw_answer(monkeypatch, tmp_path):
    invoke = AsyncMock()
    monkeypatch.setattr(run.browsecomp, "episode", invoke)
    monkeypatch.setattr(run, "preflight", AsyncMock())
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
    (root / "manifest.json").write_text(json.dumps({"tasks": ["test"], "repetitions": 1, "episodes": [],
        "config": {"retrieval_url": run.BACKEND}}))
    monkeypatch.setattr(evaluation, "load_queries", lambda _: {"test": query()})
    args = SimpleNamespace(run=root, source=source)
    evaluation.main(args)
    gold.write_text("name,value\nA,2\n")
    with pytest.raises(ValueError, match="provenance changed"):
        evaluation.main(args)


@pytest.mark.parametrize("upstream_error", [False, True])
def test_paused_scoring_preserves_later_cached_scores(monkeypatch, tmp_path, upstream_error):
    source = tmp_path / 'source'
    (source / 'widesearch_gold').mkdir(parents=True)
    (source / 'widesearch.jsonl').write_text('metadata')
    (source / 'widesearch_gold/test.csv').write_text('name,value\nA,1\n')
    root = tmp_path / 'run'
    root.mkdir()
    episodes = [{'task': 'test', 'repetition': n, 'status': 'finished', 'directory': str(n)} for n in [1, 2, 3]]
    evaluation.save(root / 'manifest.json', {'tasks': ['test'], 'repetitions': 3, 'episodes': episodes,
        'status': 'completed', 'config': {'retrieval_url': run.BACKEND}})
    for n in [1, 2, 3]:
        (root / str(n)).mkdir()
        (root / str(n) / 'answer.md').write_text('answer')
    cached = {'status': 'evaluated', 'score': 1, 'f1_by_row': 1, 'f1_by_item': 1}
    (root / 'evaluation/test-r002').mkdir(parents=True)
    evaluation.save(root / 'evaluation/test-r002/result.json', cached)
    monkeypatch.setattr(evaluation, 'load_queries', lambda _: {'test': query()})
    calls = []
    def fail(*args):
        calls.append(args)
        if upstream_error:
            return {'status': 'evaluation_error', 'score': None, 'msg': 'evaluator error: 401 AuthenticationError'}
        raise RuntimeError('AuthenticationError: 401')
    monkeypatch.setattr(evaluation, 'score', fail)
    evaluation.main(SimpleNamespace(run=root, source=source))
    summary = json.loads((root / 'evaluation/summary.json').read_text())
    assert summary['metrics']['score']['avg'] == pytest.approx(1 / 3)
    assert len(calls) == 1
    assert summary['evaluation_errors'] == 1
    assert summary['status'] == 'needs_attention'


def test_widesearch_shares_budget_free_browsecomp_harness():
    from gyms.widesearch import agents
    from gyms.browsecomp import agents as browsecomp
    assert agents.researcher is browsecomp.researcher
    assert agents.researcher_4b_thinking is browsecomp.researcher_4b_thinking
    assert agents.decomposer_sft_1_4_0 is browsecomp.decomposer_sft_1_4_0
    assert run.preflight is run.browsecomp.preflight
    assert run.parser().get_default("retrieval_url") == "http://127.0.0.1:18082"


@pytest.mark.parametrize('failure', [None, 'transient', 'permanent', 'credentials'])
def test_watch_scores_new_episodes_reuses_results_and_bounds_retries(monkeypatch, tmp_path, failure):
    import fcntl

    source = tmp_path / 'source'
    (source / 'widesearch_gold').mkdir(parents=True)
    (source / 'widesearch.jsonl').write_text('metadata')
    (source / 'widesearch_gold/test.csv').write_text('name,value\nA,1\n')
    root = tmp_path / 'run'
    root.mkdir()
    episodes = [{'task': 'test', 'repetition': n, 'status': 'finished', 'directory': str(n)} for n in [1, 2]]
    manifest = {'tasks': ['test'], 'repetitions': 2, 'episodes': episodes[:1],
                'status': 'running', 'config': {'retrieval_url': run.BACKEND}}
    evaluation.save(root / 'manifest.json', manifest)
    for n in [1, 2]:
        (root / str(n)).mkdir()
        (root / str(n) / 'answer.md').write_text(str(n))
    monkeypatch.setattr(evaluation, 'load_queries', lambda _: {'test': query()})
    calls = []

    def score(query, answer, destination):
        with (root / '.evaluation.lock').open('a') as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        calls.append(answer)
        if failure == 'credentials':
            raise RuntimeError('AuthenticationError: 401')
        if answer == '1' and (failure == 'permanent' or failure == 'transient' and calls.count('1') == 1):
            raise RuntimeError('temporary judge error')
        return {'status': 'evaluated', 'score': 1, 'f1_by_row': 1, 'f1_by_item': 1}

    def finish_generation(seconds):
        assert seconds == 60
        manifest.update(status='completed', episodes=episodes)
        evaluation.save(root / 'manifest.json', manifest)

    monkeypatch.setattr(evaluation, 'score', score)
    monkeypatch.setattr(evaluation.time, 'sleep', finish_generation)
    evaluation.main(SimpleNamespace(run=root, source=source, watch=True))
    assert calls.count('1') == {None: 1, 'transient': 2, 'permanent': 3, 'credentials': 1}[failure]
    assert calls.count('2') == (0 if failure == 'credentials' else 1)
    summary = json.loads((root / 'evaluation/summary.json').read_text())
    assert summary['status'] == ('needs_attention' if failure in ['permanent', 'credentials'] else 'completed')
