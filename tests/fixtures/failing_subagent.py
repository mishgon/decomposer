from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def fail() -> str:
    """Raise a deterministic error for the server integration test."""
    raise RuntimeError("deliberate subagent tool failure")


graph = create_agent(
    ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "fail", "args": {}, "id": "fail-call"}],
            )
        ]
    ),
    tools=[fail],
)
