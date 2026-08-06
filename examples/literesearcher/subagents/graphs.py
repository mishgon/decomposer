import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools import search, fetch


def literesearcher_4b():
    with open("prompt.txt", "r") as f:
        system_prompt = f.read()

    model = ChatOpenAI(
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        base_url=os.environ["LLM_PROXY_URL"],
        api_key=os.environ["LLM_PROXY_MASTER_KEY"],
        temperature=1.0,
        top_p=0.95,
        presence_penalty=1.5,
        use_responses_api=False,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    return create_agent(
        model=model, tools=[search, fetch], system_prompt=system_prompt
    ).with_config({"recursion_limit": 10})
