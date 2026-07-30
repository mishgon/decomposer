import os
import time
import json
import asyncio
import argparse
import logging
from contextlib import contextmanager

from langgraph_sdk import get_client
from langchain_openai import ChatOpenAI

from decomposer.core import create_decomposer_agent
from data.browsecomp_plus import load

from .verify import verify_answer
from .trace import render_decomposer_messages

# logging.basicConfig(level=logging.INFO)

DECOMPOSER_ADDITIONAL_PROMPT = """
Final response should end with 

```json
{"final_answer": "<short answer: a named entity, number, or short phrase>", "gold_sources": ["https://example.org/<page>"]}
```
"""


@contextmanager
def timed(name: str):
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        elapsed_ns = time.perf_counter_ns() - start
        print(f"{name} took {elapsed_ns / 1_000_000:.2f} ms")


async def query(client, semaphore, index, question, gt, mode):
    async with semaphore:
        is_correct = False
        try:
            if mode == "decomposer":
                question += DECOMPOSER_ADDITIONAL_PROMPT
            lg_messages = {"messages": [{"role": "user", "content": question}]}
            if mode == "direct":
                state = await client.runs.wait(
                    None,
                    "researcher",
                    input=lg_messages,
                )
            elif mode == "decomposer":
                state = await client.ainvoke(lg_messages)
                with open(f"artifacts/browsecomp_plus/{index}.md", "w") as f:
                    f.write(render_decomposer_messages(state["messages"]))
            message = state["messages"][-1]
            output = (
                message.get("content")
                if isinstance(message, dict)
                else getattr(message, "content", None)
            )
            if not isinstance(output, str) and output is not None:
                output = json.dumps(output, ensure_ascii=False)
            is_correct = await verify_answer(question, output, gt)
        except Exception as error:
            output = f"ERROR: {error}"
        return index, output, is_correct


async def main(mode, concurrency=10, limit=-1):
    assert mode in ["decomposer", "direct"]

    tasks = load()
    if limit != -1:
        tasks = tasks[:limit]

    semaphore = asyncio.Semaphore(concurrency)

    if mode == "decomposer":
        os.makedirs("artifacts/browsecomp_plus", exist_ok=True)
        model = ChatOpenAI(
            model="Qwen/Qwen3.6-35B-A3B-FP8",
            base_url=os.environ["LLM_PROXY_URL"],
            api_key=os.environ["LLM_PROXY_MASTER_KEY"],
            temperature=1.0,
            top_p=0.95,
            presence_penalty=1.5,
            max_tokens=16384,
            use_responses_api=False,
            extra_body={
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        agent = create_decomposer_agent(
            decomposer_model=model,
            subagent_types=[
                {
                    "subagent_type_id": "researcher",
                    "description": "Researcher with web search tools",
                    "assistant_id": "researcher",
                    "url": "http://127.0.0.1:2024",
                }
            ],
        )
    elif mode == "direct":
        agent = get_client(url="http://127.0.0.1:2024")

    pending = [
        query(agent, semaphore, index, question, answer, mode)
        for index, (question, answer) in enumerate(tasks, 1)
    ]

    correct = 0
    with timed("run"):
        for completed in asyncio.as_completed(pending):
            index, output, is_correct = await completed
            correct += 1 if is_correct else 0
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            print(f"[{index}]", "🟢" if is_correct else "🔴", output, sep="\t")
        print(f"Total: {correct}/{len(tasks)}\tAcc: {100 * correct / len(tasks)}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrowseComp-Plus benchmark")
    parser.add_argument("mode", choices=["decomposer", "direct"])
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    asyncio.run(main(args.mode, args.concurrency, args.limit))
