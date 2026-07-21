from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


def qwen3_6_35b_a3b_fp8_thinking():
    model = ChatOpenAI(
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        base_url="http://127.0.0.1:8019/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        presence_penalty=1.5,
        max_tokens=32768,
        use_responses_api=False,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    return create_agent(model=model, tools=[], system_prompt="You are helpful assistant")
