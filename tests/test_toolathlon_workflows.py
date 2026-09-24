import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from gyms.toolathlon_gym import parallel, run as gym
from evals.toolathlon_gym.summary import summarize
from tests.fixtures.conversation_subagent import ConversationTestModel
from tests.test_subagent_server import subagent_server


@pytest.mark.parametrize("harness", ["react", "decomposer"])
def test_both_harnesses_execute_real_tool_loop(subagent_server, monkeypatch, harness):
    monkeypatch.setattr(gym, "SUBAGENT_TYPE_ID", "worker")
    episode_id = "coverage-task-r001-" + str(uuid4())
    agent, config = gym.make_agent(harness, ConversationTestModel(is_decomposer=True),
                          subagent_server, "fixture", episode_id, InMemorySaver())
    state, error = asyncio.run(gym.invoke_and_capture(agent,
        {"messages": [{"role": "user", "content": "Remember 42."}]},
        config, 30))
    assert error is None
    messages = gym.serialize_messages(state["messages"])
    last = messages[-1].get("data", messages[-1])
    assert json.loads(last["content"]) == ["Remember 42."]
    assert any(m.get("data", m).get("tool_calls") for m in messages)
    if harness == "decomposer":
        assert len(state["subagents"]) == len(state["subagent_runs"]) == 1
        assert next(iter(state["subagent_runs"].values()))["messages"]


@pytest.mark.parametrize("harness", ["react", "decomposer"])
def test_fixed_parallel_execution_and_raw_artifacts(tmp_path, monkeypatch, harness):
    monkeypatch.setattr(gym, "TOOLATHLON_ROOT", tmp_path / "gym")
    for task in ("a", "b"):
        (gym.TOOLATHLON_ROOT / "tasks/finalpool" / task).mkdir(parents=True)
    seen = []
    def execute(command, directory, timeout, stop):
        args = gym.create_parser().parse_args(command[2:])
        seen.append((args.task, args.repetition, args.harness))
        result = args.evals_dir / args.task / args.episode_id / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps({"pass": args.task == "a"}))
        return 0, False
    monkeypatch.setattr(parallel, "execute", execute)
    monkeypatch.setattr(gym, "start_vllm", lambda **kw: pytest.fail("Hosted run started vLLM"))
    args = gym.create_parser().parse_args(["--all", "--harness", harness,
        "--subagent-base-url", "https://router.test/v1", "-n", "3", "--concurrency", "2",
        "--output-dir", str(tmp_path / "raw")])
    root = parallel.run(args)
    assert len(seen) == 6
    assert all(item[2] == harness for item in seen)
    summary = summarize(root)
    assert summary["metrics"] == {"pass@1": 0.5, "pass@3": 0.5, "pass^3": 0.5}


def test_metrics_count_missing_and_infra_failures(tmp_path):
    episodes = []
    for repetition in (1, 2, 3):
        eid = str(repetition)
        episodes.append({"task": "a", "repetition": repetition, "episode_id": eid,
                         "returncode": 0 if repetition < 3 else 1})
        path = tmp_path / "evals/a" / eid / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"pass": true}')
    (tmp_path / "manifest.json").write_text(json.dumps({"tasks": ["a", "b"],
        "repetitions": 3, "episodes": episodes}))
    result = summarize(tmp_path, denominator=4)
    assert result["successful_attempts"] == 2
    assert result["metrics"]["pass@1"] == pytest.approx(2 / 12)
    assert result["metrics"]["pass@3"] == 0.25
    assert result["metrics"]["pass^3"] == 0


def test_gym_does_not_import_workflows():
    root = Path(gym.__file__).parent
    for path in root.rglob("*.py"):
        source = path.read_text()
        assert "from sft." not in source
        assert "from evals." not in source


def test_partial_state_is_saved_on_failure():
    class Agent:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("provider unavailable")
        async def aget_state(self, config):
            from types import SimpleNamespace
            return SimpleNamespace(values={"messages": [{"type": "ai", "content": "partial"}]})
    state, error = asyncio.run(gym.invoke_and_capture(Agent(), {}, {}, 1))
    assert str(error) == "provider unavailable"
    assert gym.serialize_messages(state["messages"])[0]["content"] == "partial"
