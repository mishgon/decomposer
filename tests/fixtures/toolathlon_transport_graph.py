"""Deterministic reproduction of a model emitting an out-of-range page number."""
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import tool
from graph import transport_safe_response


class InvalidArgumentsModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "invalid-arguments-fixture"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[
            {"id": "bad-page", "name": "page", "args": {"number": 10**21}}]))])

    async def _agenerate(self, messages, **kwargs):
        return self._generate(messages)


@tool
def page(number: int) -> str:
    """Read a page."""
    raise AssertionError("Unsafe arguments must never execute")


agent = create_agent(InvalidArgumentsModel(), tools=[page], middleware=[transport_safe_response])
