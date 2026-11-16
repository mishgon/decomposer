import asyncio
from contextlib import asynccontextmanager
import json
from unittest.mock import AsyncMock
from types import SimpleNamespace

from langchain_core.messages import AIMessage, message_to_dict
import pytest

from gyms.masbench import RULES, task_input
from gyms.masbench import agents, run
from gyms.masbench.evaluate import boxed_answers, evaluate, final_output
from gyms.masbench.prepare import convert


@pytest.mark.parametrize("output,gold,passed", [
    (r"\boxed{0}", "0", True),
    (r"\boxed{23}", "0", False),
    (r"\boxed{12345}", "12345", True),
    (r"\boxed{1} then \boxed{2}", "1<<horizon>>2", True),
    (r"\boxed{2} then \boxed{1}", "1<<horizon>>2", False),
    (r"\boxed{1}", "1<<horizon>>2", False),
    (r"\boxed{1}\boxed{2}\boxed{2}", "1<<horizon>>2", False),
    (r"\boxed{\frac{4}{2}}", "2", True),
    (r"\boxed{1} correction: \boxed{2}", "2", True),
    ("2", "2", True),
    ("", "0", False),
])
def test_scoring(output, gold, passed):
    assert evaluate(output, gold)["passed"] is passed


def test_boxes_and_reasoning_exclusion():
    assert boxed_answers(r"\boxed{broken \boxed{2}") == ["2"]
    assert final_output([message_to_dict(AIMessage(
        content=r"\boxed{1}", additional_kwargs={"reasoning_content": r"\boxed{2}"},
    ))]) == r"\boxed{1}"
    assert final_output([message_to_dict(AIMessage(
        content=r"\boxed{1}", tool_calls=[{"id": "c", "name": "wait", "args": {}}],
    ))]) == ""


def test_no_gold_or_graph_in_model_input():
    row = {"prompt_json": json.dumps([{"role": "user", "content": "Original question"}]),
           "value": "3", "reward_model_json": json.dumps({"ground_truth": "SECRET"}),
           "extra_info_json": "SECRET_GRAPH"}
    task = convert(row, "depth", 0)
    assert task_input(task) == {"messages": [{"role": "user", "content": RULES + "\nOriginal question"}]}
    assert "SECRET" not in json.dumps(task_input(task))


def test_agent_prompt_and_model_separation(monkeypatch):
    monkeypatch.setenv("MASBENCH_BASE_URL", "http://base/v1")
    monkeypatch.setenv("MASBENCH_SFT_URL", "http://sft/v1")
    monkeypatch.setenv("MASBENCH_AGENT_URL", "http://workers")
    monkeypatch.setattr(agents, "create_model", lambda model, **kwargs: (model, kwargs))
    monkeypatch.setattr(agents, "create_agent", lambda *args, **kwargs: (args, kwargs))
    monkeypatch.setattr(agents, "create_decomposer_agent", lambda **kwargs: kwargs)
    args, single = agents.single()
    assert args[0] == (agents.AGENTS["single"], {"base_url": "http://base/v1"})
    assert single["tools"] == []
    assert single["system_prompt"] == agents.AGENT_SYSTEM_PROMPT + "\n\n" + RULES
    for factory, url in ((agents.base_manager, "http://base/v1"), (agents.sft_manager, "http://sft/v1")):
        manager = factory()
        assert manager["decomposer_model"][1] == {"base_url": url}
        assert "system_prompt" not in manager
        assert manager["agent_types"][0]["assistant_id"] == "single"
        assert manager["agent_run_budget_seconds"] is None


@pytest.mark.parametrize("error", [None, TimeoutError("stopped")])
def test_episode_preserves_raw_output_but_does_not_pass_timeout(monkeypatch, tmp_path, error):
    task = {"task_id": "depth-0000", "axis": "depth", "value": "2", "question": "Q", "ground_truth": "1"}
    (tmp_path / "task.json").write_text(json.dumps(task))

    @asynccontextmanager
    async def server(*args, **kwargs):
        yield "http://fake"

    capture = AsyncMock(return_value=({"messages": [message_to_dict(AIMessage(content=r"\boxed{1}"))]}, error))
    monkeypatch.setattr(run, "agent_server", server)
    monkeypatch.setattr(run, "invoke_and_capture", capture)
    args = run.parser().parse_args(["--episode", "--agent", "single", "--output-dir", str(tmp_path)])
    asyncio.run(run.episode(args))
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["evaluation"]["passed"] is (error is None)
    assert (tmp_path / "answer.txt").read_text() == r"\boxed{1}"
    assert capture.call_args.args[2] == task_input(task)
    assert (tmp_path / "trace.json").exists()
    assert (tmp_path / "usage.json").exists()


@pytest.mark.parametrize("interrupt_error", [
    "CancelledError()", "Exception('UserInterrupt: User interrupted the run')",
    "RuntimeError('Agent run interrupted')",
])
def test_queue_stops_after_repeated_crashes_and_records_failed_attempts(monkeypatch, tmp_path, interrupt_error):
    data = tmp_path / "tasks.jsonl"
    data.write_text("\n".join(json.dumps({
        "task_id": f"depth-{index:04d}", "axis": "depth", "value": "2",
        "question": "Q", "ground_truth": "1",
    }) for index in range(5)))
    monkeypatch.setenv("MASBENCH_BASE_URL", "http://base/v1")
    monkeypatch.setenv("MASBENCH_SFT_URL", "http://sft/v1")
    monkeypatch.setattr(run, "preflight", AsyncMock())
    monkeypatch.setattr(run.asyncio, "create_subprocess_exec", AsyncMock(
        return_value=SimpleNamespace(returncode=1, wait=AsyncMock(return_value=1))))
    output = tmp_path / "run"
    args = run.parser().parse_args(["--all", "--data", str(data), "--output-dir", str(output),
                                   "--agents", "single", "--concurrency", "1"])
    with pytest.raises(RuntimeError, match="needs_attention"):
        asyncio.run(run.main(args))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "needs_attention"
    assert len(manifest["episodes"]) == 3
    assert all(row["evaluation"]["score"] == 0 for row in manifest["episodes"])

    # Recover a finished result not yet checkpointed by the coordinator; retry
    # an interrupted attempt, without retrying any of the three recorded errors.
    def result(index, error=None):
        return {"task_id": f"depth-{index:04d}", "axis": "depth", "value": "2",
                "agent": "single", "repetition": 1, "error": error,
                "status": "error" if error else "finished",
                "evaluation": {"passed": error is None, "score": int(error is None)}}

    for index, error in ((3, None), (4, interrupt_error)):
        directory = output / "single" / f"depth-{index:04d}-r001"
        directory.mkdir()
        (directory / "result.json").write_text(json.dumps(result(index, error)))
    # A coordinator may already have recorded the SDK's interruption response.
    manifest["episodes"].append(result(4, interrupt_error))
    (output / "manifest.json").write_text(json.dumps(manifest))

    args.resume = True
    args.timeout += 1
    with pytest.raises(ValueError, match="timeout_seconds"):
        asyncio.run(run.main(args))
    args.timeout -= 1

    async def spawn(*command, **kwargs):
        directory = run.Path(command[command.index("--output-dir") + 1])
        assert directory.name == "depth-0004-r001"
        (directory / "result.json").write_text(json.dumps(result(4)))
        return SimpleNamespace(returncode=0, wait=AsyncMock(return_value=0))

    mock_spawn = AsyncMock(side_effect=spawn)
    monkeypatch.setattr(run.asyncio, "create_subprocess_exec", mock_spawn)
    args.concurrency = 2
    with pytest.raises(RuntimeError, match="needs_attention"):
        asyncio.run(run.main(args))
    mock_spawn.assert_awaited_once()
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["episodes"]) == 5
    assert sum(row["status"] == "error" for row in manifest["episodes"]) == 3
    assert manifest["resumptions"][0]["previous_concurrency"] == 1
    assert manifest["resumptions"][0]["concurrency"] == 2
    archived = list((output / "interrupted/single").glob("*/result.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text())["error"] == interrupt_error
    assert manifest["resumptions"][0]["interrupted_results"][0]["error"] == interrupt_error
