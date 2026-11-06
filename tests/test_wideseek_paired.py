import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from evals.wideseek.paired import LEAD_RECURSION_LIMIT, RESEARCHER_RECURSION_LIMIT
from gyms.wideseek import agents
from gyms.wideseek.runtime import init_budget


class FakeModel(FakeMessagesListChatModel):
    def bind_tools(self, *args, **kwargs):
        return self


@pytest.mark.parametrize('role,turns,limit', [
    ('lead', 10, LEAD_RECURSION_LIMIT), ('researcher', 20, RESEARCHER_RECURSION_LIMIT)])
@pytest.mark.parametrize('short', [False, True])
def test_paper_turn_path_and_recursion_boundary(tmp_path, monkeypatch, role, turns, limit, short):
    policy = FakeModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'invalid', 'args': {}, 'id': str(i)}])
        for i in range(turns - 1)] + [AIMessage(content='done')])
    monkeypatch.setattr(agents, 'model', lambda _: policy)
    graph = (agents.decomposer(policy, InMemorySaver(), 'http://unused')
             if role == 'lead' else agents.researcher())
    path = tmp_path / 'episode'
    init_budget(path)
    events = []

    async def run():
        async for event in graph.astream({'messages': [('user', 'test')]},
                {'recursion_limit': limit - short, 'configurable': {'thread_id': 'test'}},
                context={'directory': str(path)}, stream_mode='updates'):
            events.extend(event)

    if short:
        with pytest.raises(GraphRecursionError):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert events.count('model') == turns
