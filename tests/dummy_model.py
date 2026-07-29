from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class DummyModel(BaseChatModel):
    response: str = Field(default="dummy")
    tool_calls: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "dummy"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, **kwargs):
        tool_calls = (
            []
            if any(isinstance(message, ToolMessage) for message in messages)
            else self.tool_calls
        )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="" if tool_calls else self.response,
                        tool_calls=tool_calls,
                    )
                )
            ]
        )
