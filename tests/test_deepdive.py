import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gyms.deepdive.prepare import normalize
from gyms.wideseek.prepare import agent_input


def test_normalization_keeps_split_identity_and_hides_reference():
    row = {"id": 7, "question": "Which city?", "answer": "Secret answer",
           "conversations": [{"role": "assistant", "content": "Secret solution"}]}
    rl = normalize(row, "qa_rl")
    sft = normalize(row, "qa_sft")
    assert rl["task_id"] != sft["task_id"]
    assert rl["metadata"]["original_id"] == 7
    assert rl["question_sha256"] == sft["question_sha256"]
    assert rl["answer"] == row["answer"]
    assert rl["unique_columns"] == []
    prompt = json.dumps(agent_input(rl))
    assert row["question"] in prompt
    assert "Secret" not in prompt
    assert "conversations" not in rl["metadata"]


def test_depth_and_deepdive_share_answer_format():
    from gyms.wideseek.prepare import normalize as normalize_wideseek

    row = {"id": 7, "question": "Which city?", "answer": "Paris"}
    depth = normalize_wideseek(row, "depth", 7)
    deepdive = normalize(row, "qa_rl")
    assert depth["unique_columns"] == deepdive["unique_columns"] == []
    assert agent_input(depth) == agent_input(deepdive)
    assert agent_input(depth)["messages"][0]["content"].endswith(
        r"you have to wrap your final answer in \boxed{}.")


def test_preparation_retains_duplicate_questions_and_pins_source(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gyms.deepdive import prepare

    raw = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"id": 1, "question": "Same question", "answer": "A"},
        {"id": 2, "question": "Same question", "answer": "A"},
    ]), raw)
    urls = []

    def download(url, **kwargs):
        urls.append(url)
        return raw.open("rb")

    monkeypatch.setattr(prepare, "urlopen", download)
    monkeypatch.setitem(prepare.SPLITS, "qa_rl", 2)
    output = tmp_path / "prepared"
    manifest = prepare.prepare("qa_rl", output)
    assert manifest["examples"] == 2
    assert manifest["unique_questions"] == 1
    assert manifest["filtered"] is False
    assert prepare.REVISION in urls[0]
    assert len((output / "tasks.jsonl").read_text().splitlines()) == 2
    assert (output / "source.parquet").read_bytes() == raw.read_bytes()


@pytest.mark.parametrize("source", ["depth", "deepdive"])
@pytest.mark.parametrize("failure", [False, True])
def test_qa_judge_failure_stays_unscored(tmp_path, monkeypatch, failure, source):
    from gyms.wideseek import evaluate
    from langchain_core.messages import AIMessage

    judge = AsyncMock()
    if failure:
        judge.ainvoke.side_effect = RuntimeError("judge unavailable")
    else:
        judge.ainvoke.return_value = AIMessage(content="Correct")
    monkeypatch.setattr(evaluate, "model", lambda _: judge)
    monkeypatch.setattr(evaluate, "close_model", AsyncMock())
    row = {"id": 1, "question": "Which city?", "answer": "Paris"}
    from gyms.wideseek.prepare import normalize as normalize_wideseek
    task = normalize(row, "qa_rl") if source == "deepdive" else normalize_wideseek(row, "depth", 1)
    result = asyncio.run(evaluate.evaluate(task, r"\boxed{Paris}", tmp_path))
    assert result["metric"] == "qa_accuracy"
    assert result["score"] == (None if failure else 1.0)
    assert result["status"] == ("judge_error" if failure else "scored")
    assert len(list((tmp_path / "judge_calls").glob("*.json"))) == 1


def test_launcher_uses_deepdive_data_and_cleans_workers_on_failure(tmp_path, monkeypatch):
    from sft.deepdive import launch

    args = launch.create_parser().parse_args([
        "--agent", "decomposer", "--output", str(tmp_path),
    ])
    assert "deepdive" in args.data.parts
    assert args.data.parent.name == "all-qa"
    assert args.concurrency == 128
    assert args.limit is None and args.n == 1
    assert args.episode_timeout == 2700
    assert args.output_tokens is None
    assert not args.adaptive
    events = []

    @asynccontextmanager
    async def workers(config, **kwargs):
        assert config.parent.name == "deepdive"
        assert kwargs["n_jobs_per_worker"] == 512
        events.append("started")
        try:
            yield "http://worker"
        finally:
            events.append("stopped")

    async def collect(args):
        assert args.worker_url == "http://worker"
        raise RuntimeError("collection failed")

    monkeypatch.setattr("sft.wideseek.launch.agent_server", workers)
    monkeypatch.setattr(launch, "collect", collect)
    with pytest.raises(RuntimeError, match="collection failed"):
        asyncio.run(launch.main(args))
    assert events == ["started", "stopped"]


def test_deepdive_worker_uses_wideseek_retrieval_agent():
    root = Path(__file__).resolve().parents[1]
    graphs = json.loads((root / "gyms/deepdive/langgraph.json").read_text())["graphs"]
    assert graphs == {"researcher": "gyms.wideseek.agents:researcher"}


def test_collection_pool_mixes_both_qa_splits(tmp_path):
    from random import Random
    from sft.deepdive.prepare import DEFAULT_INPUTS, prepare

    assert {p.parent.name for p in DEFAULT_INPUTS} == {"qa_rl", "qa_sft"}
    paths, original = [], []
    for split in ("qa_rl", "qa_sft"):
        rows = [normalize({"id": i, "question": f"Question {i}", "answer": "Reference"}, split)
                for i in range(10)]
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths.append(path)
        original.extend(rows)
    manifest = prepare(paths, tmp_path / "pool")
    actual = [json.loads(line) for line in (tmp_path / "pool/tasks.jsonl").read_text().splitlines()]
    Random(42).shuffle(original)
    assert actual == original
    assert len({row["task_id"] for row in actual}) == 20
    assert {row["metadata"]["split"] for row in actual[:5]} == {"qa_rl", "qa_sft"}
    assert manifest["shuffle_seed"] == 42
    assert manifest["sources"] == {"deepdive-qa_rl": 10, "deepdive-qa_sft": 10}
    for row in actual:
        assert "Reference" not in json.dumps(agent_input(row))
    with pytest.raises(ValueError, match="unique task IDs"):
        prepare([paths[0], paths[0]], tmp_path / "duplicate")
