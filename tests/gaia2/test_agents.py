import pytest

from gyms.gaia2 import agents
from gyms.gaia2.tools import EpisodeContext


@pytest.mark.parametrize("worker", ["vllm/qwen_3_5_4b_thinking", "vllm/timur_4b_non_thinking"])
def test_manager_uses_core_factory_and_server_checkpointing(tmp_path, monkeypatch, worker):
    for name, value in {
        "GAIA2_DECOMPOSER_MODEL_ID": "fake",
        "GAIA2_AGENT_MODEL_ID": worker,
        "GAIA2_DECOMPOSER_BASE_URL": "http://manager/v1",
        "GAIA2_AGENT_SERVER_URL": "http://server",
        "GAIA2_AGENT_RUN_BUDGET_SECONDS": "17",
        "GAIA2_AGENT_SHUTDOWN_GRACE_SECONDS": "23",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        agents,
        "create_model",
        lambda model_id, **kwargs: (model_id, kwargs["base_url"]),
    )
    monkeypatch.setattr(agents, "create_decomposer_agent", lambda **kwargs: kwargs)
    result = agents.decomposer()
    assert result["decomposer_model"] == ("fake", "http://manager/v1")
    expected = [
        {
            "agent_type_id": "qwen_3_5_4b_thinking",
            "description": "Qwen3.5-4B thinking agent equipped with all the required tools.",
            "assistant_id": "qwen_3_5_4b_thinking",
            "url": "http://server",
        }
    ]
    if worker == "vllm/timur_4b_non_thinking":
        expected[0].update(
            agent_type_id="timur_4b_non_thinking", assistant_id="timur_4b_non_thinking",
            description="Timur Qwen3.5-4B non-thinking agent equipped with all the required tools.",
        )
        assert agents.timur_4b_non_thinking is agents.qwen_3_5_4b_thinking
        from gyms.gaia2.run import build_parser
        assert build_parser().parse_args(["--agent", "timur_4b_non_thinking"]).agent == "timur_4b_non_thinking"
    assert result["agent_types"] == expected
    assert result["agent_run_budget_seconds"] == 17
    assert result["agent_shutdown_grace_seconds"] == 23
    assert result["context_schema"] is EpisodeContext
    assert "checkpointer" not in result


def test_trace_terminal_statuses_match_core():
    from decomposer.core import TERMINAL_STATUSES
    from gyms.gaia2.are import TERMINAL_STATUSES as TRACE_STATUSES

    assert TRACE_STATUSES == TERMINAL_STATUSES


@pytest.mark.parametrize("budget", [None, 60])
def test_shared_agent_graph_applies_budget_only_when_supplied(monkeypatch, budget):
    import asyncio

    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage

    from decomposer.run_budget import RunBudgetMiddleware

    observed = []
    original = RunBudgetMiddleware.before_agent

    def before_agent(self, state, runtime):
        result = original(self, state, runtime)
        observed.append(result)
        return result

    monkeypatch.setattr(RunBudgetMiddleware, "before_agent", before_agent)
    monkeypatch.setattr(
        agents,
        "_model",
        lambda: FakeMessagesListChatModel(responses=[AIMessage(content="Done")]),
    )
    monkeypatch.setattr(agents, "_worker_tools", lambda context, consumer: [])
    context = {"worker_system_prompt": "Test prompt"}
    if budget is not None:
        context.update(
            agent_run_budget_seconds=budget, agent_shutdown_grace_seconds=60
        )
    result = asyncio.run(
        agents.qwen_3_5_4b_thinking.ainvoke(
            {"messages": [{"role": "user", "content": "Task"}]},
            context=context,
        )
    )
    assert result["messages"][-1].content == "Done"
    assert (observed[0]["shutdown_at"] is None) == (budget is None)
    assert (observed[0]["timeout_at"] is None) == (budget is None)
