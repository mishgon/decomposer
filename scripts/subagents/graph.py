from decomposer.chat_vllm import ChatVLLM
from decomposer.prompts import SUBAGENT_SYSTEM_PROMPT
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph


REQUEST_TIMEOUT_SECONDS = 300.0


def qwen_3_5_4b_non_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="Qwen/Qwen3.5-4B",
        base_url="http://127.0.0.1:8024/v1",
        api_key="EMPTY",
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=False,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_agent(model=model, tools=[], system_prompt=SUBAGENT_SYSTEM_PROMPT)
