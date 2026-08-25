from __future__ import annotations

from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from responses_api_agents.decomposer_agent.subagents.graph import (
    SYSTEM_PROMPT,
    NeMoGymSubagentMiddleware,
)

from decomposer.chat_vllm import ChatVLLM
from gyms.qwen_sampling import qwen35_general_sampling

REQUEST_TIMEOUT_SECONDS = 300.0


def qwen35_4b_non_thinking() -> CompiledStateGraph:
    sampling = qwen35_general_sampling(thinking=False)
    model = ChatVLLM(
        model="Qwen/Qwen3.5-4B",
        base_url="http://127.0.0.1:8025/v1",
        api_key="EMPTY",
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        presence_penalty=sampling.presence_penalty,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=False,
        extra_body={
            **sampling.extra_body,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_agent(
        model=model,
        tools=[],
        middleware=[NeMoGymSubagentMiddleware()],
        system_prompt=SYSTEM_PROMPT,
    )
