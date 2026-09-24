import json

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from decomposer import create_decomposer_agent


class ConversationTestModel(BaseChatModel):
    is_decomposer: bool = False

    @property
    def _llm_type(self) -> str:
        return "conversation-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prompts = [message.content for message in messages if isinstance(message, HumanMessage)]
        if not self.is_decomposer:
            if isinstance(messages[-1], HumanMessage):
                response = AIMessage(content="", tool_calls=[{
                    "name": "echo", "args": {"text": prompts[-1]}, "id": f"echo_{len(prompts)}",
                }])
            else:
                response = AIMessage(content=json.dumps(prompts))
        else:
            results = [
                json.loads(message.content)
                for message in messages
                if isinstance(message, ToolMessage)
            ]
            last_result = results[-1] if isinstance(messages[-1], ToolMessage) else None
            if isinstance(last_result, list):
                assert len(last_result) == 1
                assert last_result[0]["status"] == "responded"
                response = AIMessage(content=last_result[0]["response"])
            else:
                children = [
                    result["subagent_id"] for result in results
                    if isinstance(result, dict) and "subagent_run_id" not in result
                ]
                if not children:
                    name, args = "new", {"subagent_type_id": "worker"}
                elif last_result is not None and "subagent_run_id" in last_result:
                    name, args = "wait", {}
                else:
                    name, args = "run", {
                        "subagent_id": children[0],
                        "prompt": prompts[-1],
                    }
                response = AIMessage(content="", tool_calls=[{
                    "name": name, "args": args, "id": f"call_{len(messages)}",
                }])
        return ChatResult(generations=[ChatGeneration(message=response)])


@tool
def echo(text: str) -> str:
    """Return the supplied text."""
    return text


worker = create_agent(ConversationTestModel(), tools=[echo])
root = create_decomposer_agent(
    decomposer_model=ConversationTestModel(is_decomposer=True),
    subagent_types=[{"subagent_type_id": "worker", "assistant_id": "worker", "description": "Worker"}],
)
