import asyncio
import importlib.util
import json
import socket

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from decomposer.agent_server import agent_server
from gyms.synth import ROOT
from gyms.synth.executor import execute, grade, initialize, execute_saved
from gyms.synth.run import episode
from gyms.synth.task import make_task, reference_plans


def test_templates_are_reproducible_and_disjoint():
    assert make_task("train", 0) == make_task("train", 0)
    train = [make_task("train", i) for i in range(16)]
    val = [make_task("eval", i) for i in range(8)]
    assert not ({n for t in train for n in t["initial"]} & {n for t in val for n in t["initial"]})
    assert len({t["prompt"] for t in train + val}) == 24


def test_reference_swap_and_verification(tmp_path):
    task = make_task("train", 0)
    path = tmp_path / "episode"
    initialize(path, task)
    for i, plan in enumerate(reference_plans(task)):
        report = execute_saved(path, plan, f"worker-{i}", f"run-{i}")
        assert report.count("COPIED") == 3
    state = json.loads((path / "workspace.json").read_text())
    assert grade(task, state, {})["score"] == 1
    assert grade(task, state, {})["copies"] == 6
    for a, b in task["pairs"]:
        assert execute(state, f"CHECK {a} {b}\nCHECK {b} {a}").count("PASS") == 2


def test_destructive_copy_loses_information_and_invalid_input_is_atomic():
    task = make_task("eval", 0)
    state = {"initial": task["initial"], "files": task["initial"].copy(), "events": []}
    a, b = task["pairs"][0]
    execute(state, f"COPY {a} {b}\nCOPY {b} {a}")
    assert grade(task, state, {})["score"] == .25
    before = state["files"].copy()
    for prompt in (f"Please swap {a} and {b}", f"COPY {a} ../../secret", f"COPY {a} {b}\nrm -rf /", f"COPY {a} aaaaaaaa.txt"):
        assert execute(state, prompt).startswith(("INVALID_COMMAND", "UNKNOWN_FILE"))
        assert state["files"] == before


class PlanModel(FakeMessagesListChatModel):
    plans: list[str]
    checks: str

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        outputs = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
        def call(name, identifier, **arguments):
            return {"name": name, "id": identifier, "args": arguments}
        if "new0" not in outputs:
            calls = [call("new", f"new{i}", agent_type_id="file_worker") for i in range(2)]
        elif "run0" not in outputs:
            calls = [call("run", f"run{i}", agent_id=json.loads(outputs[f"new{i}"])["agent_id"], prompt=self.plans[i]) for i in range(2)]
        elif "wait0" not in outputs:
            calls = [call("wait", "wait0")]
        elif "reviewer" not in outputs:
            calls = [call("new", "reviewer", agent_type_id="file_worker")]
        elif "review" not in outputs:
            calls = [call("run", "review", agent_id=json.loads(outputs["reviewer"])["agent_id"], prompt=self.checks)]
        elif "wait1" not in outputs:
            calls = [call("wait", "wait1")]
        else:
            calls = []
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="" if calls else "Done", tool_calls=calls))])


def test_real_harness_and_deterministic_workers(monkeypatch, tmp_path):
    if importlib.util.find_spec("langgraph_cli") is None:
        pytest.skip("LangGraph CLI not installed")
    monkeypatch.setenv("PYTHONPATH", f"{ROOT / 'src'}:{ROOT}")
    monkeypatch.setenv("SYNTH_ARTIFACT_ROOT", str(tmp_path))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    task = make_task("eval", 0)
    policy = PlanModel(responses=[], plans=reference_plans(task), checks="\n".join(
        f"CHECK {a} {b}\nCHECK {b} {a}" for a, b in task["pairs"]))
    async def smoke():
        async with agent_server(ROOT / "gyms/synth/langgraph.json", port=port) as url:
            return await episode(task, policy, tmp_path / "episode", url)
    result = asyncio.run(smoke())
    assert result["passed"], result
    assert result["copies"] == 6
    assert result["parallel"]
    assert not result["cleanup_errors"]
    assert (tmp_path / "episode/trace.html").exists()


def test_prepared_data_has_no_answers_in_prompts(tmp_path):
    import pandas as pd
    from opd.synth.prepare import prepare
    output = tmp_path / "data"
    prepare(output)
    train = pd.read_parquet(output / "train.parquet")
    val = pd.read_parquet(output / "evaluation.parquet")
    assert len(train) == 128
    assert len(val) == 12
    assert set(val.data_source) == {"synth/train_probe", "synth/heldout"}
    assert "expected" not in train.columns
