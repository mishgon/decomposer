from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


def gemma_4_4b_thinking():
    model = ChatOpenAI(
        model="google/gemma-4-E4B-it",
        base_url="http://127.0.0.1:8021/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        timeout=300.0,
        max_retries=0,
        disable_streaming=True,
        use_responses_api=False,
        extra_body={
            "top_k": 64,
        },
    )
    return create_agent(
        model=model,
        tools=[],
        system_prompt="You are a helpful assistant.",
    )
