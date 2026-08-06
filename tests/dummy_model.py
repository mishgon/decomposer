from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class DummyModel(BaseChatModel):
    response: str = Field(default="dummy")
    tool_calls: list[dict] = Field(default_factory=list)
    responses: list[AIMessage] = Field(default_factory=list)
    response_index: int = 0
    seen_messages: list[list] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "dummy"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_messages.append([message.model_copy(deep=True) for message in messages])
        if self.responses:
            if self.response_index >= len(self.responses):
                raise AssertionError("DummyModel received more calls than configured responses.")
            message = self.responses[self.response_index].model_copy(deep=True)
            self.response_index += 1
            return ChatResult(generations=[ChatGeneration(message=message)])

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
