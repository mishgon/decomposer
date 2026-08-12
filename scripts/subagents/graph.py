from decomposer.chat_vllm import ChatVLLM
from decomposer.prompts import SUBAGENT_SYSTEM_PROMPT
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph


REQUEST_TIMEOUT_SECONDS = 300.0


def _create_subagent(model_id: str, port: int, *, thinking: bool) -> CompiledStateGraph:
    extra_body = {"top_k": 64}
    if not thinking:
        extra_body.update(
            {
                "include_reasoning": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )

    model = ChatVLLM(
        model=model_id,
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=thinking,
        extra_body=extra_body,
    )
    return create_agent(model=model, tools=[], system_prompt=SUBAGENT_SYSTEM_PROMPT)


def gemma_4_2b_thinking() -> CompiledStateGraph:
    return _create_subagent("google/gemma-4-E2B-it", 8020, thinking=True)


def gemma_4_2b_non_thinking() -> CompiledStateGraph:
    return _create_subagent("google/gemma-4-E2B-it", 8020, thinking=False)


def gemma_4_4b_thinking() -> CompiledStateGraph:
    return _create_subagent("google/gemma-4-E4B-it", 8021, thinking=True)


def gemma_4_4b_non_thinking() -> CompiledStateGraph:
    return _create_subagent("google/gemma-4-E4B-it", 8021, thinking=False)


def gemma_4_12b_thinking() -> CompiledStateGraph:
    return _create_subagent("google/gemma-4-12B-it", 8022, thinking=True)


def gemma_4_12b_non_thinking() -> CompiledStateGraph:
    return _create_subagent("google/gemma-4-12B-it", 8022, thinking=False)


def gemma_4_26b_a4b_thinking() -> CompiledStateGraph:
    return _create_subagent("google/gemma-4-26B-A4B-it", 8023, thinking=True)


def gemma_4_26b_a4b_non_thinking() -> CompiledStateGraph:
    return _create_subagent("google/gemma-4-26B-A4B-it", 8023, thinking=False)
