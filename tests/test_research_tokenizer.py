"""Exercise the benchmark researcher with its real context tokenizer."""
import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


@pytest.mark.integration
@pytest.mark.parametrize("gym", ["browsecomp", "widesearch"])
def test_researcher_finishes_with_real_tokenizer(monkeypatch, tmp_path, gym):
    from gyms.browsecomp import agents
    from gyms.widesearch import agents as widesearch_agents

    class Model(FakeMessagesListChatModel):
        model_name: str = "test"

        def bind_tools(self, tools, **kwargs):
            return self

    model = Model(responses=[AIMessage(content="", tool_calls=[{
        "id": "done", "name": "finish", "args": {"answer": "Unknown", "sources": []},
    }])])
    monkeypatch.setattr(agents, "create_model", lambda _: model)
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    factory = agents if gym == "browsecomp" else widesearch_agents
    result = asyncio.run(factory.researcher_occ_non_thinking().ainvoke(
        {"messages": [{"role": "user", "content": "Report that the answer is unknown."}]},
        context={"directory": str(tmp_path)},
    ))
    assert result["end_status"] == "finished"
    assert result["messages"][-1].content == "Unknown"
    assert (tmp_path / "model_calls.jsonl").is_file()
