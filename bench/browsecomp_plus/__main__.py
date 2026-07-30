import os
import time
import json
import asyncio
import argparse
from contextlib import contextmanager

from langgraph_sdk import get_client
from langchain_openai import ChatOpenAI

from decomposer.core import create_decomposer_agent
from .dataset import load
from .verify import verify_answer


@contextmanager
def timed(name: str):
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        elapsed_ns = time.perf_counter_ns() - start
        print(f"{name} took {elapsed_ns / 1_000_000:.2f} ms")


async def query(client, semaphore, index, question, gt):
    async with semaphore:
        is_correct = False
        try:
            lg_messages = {"messages": [{"role": "user", "content": question}]}
            state = await client.runs.wait(
                None,
                "researcher",
                input=lg_messages,
            )
            output = state["messages"][-1]["content"]
            is_correct = await verify_answer(question, output, gt)
        except Exception as error:
            output = f"ERROR: {error}"
        return index, output, is_correct


async def direct(
    tasks: list[tuple[str, str]],
    concurrency: int,
) -> None:
    client = get_client(url="http://127.0.0.1:2024")
    semaphore = asyncio.Semaphore(concurrency)
    pending = [
        query(client, semaphore, index, question, answer)
        for index, (question, answer) in enumerate(tasks, 1)
    ]
    correct = 0
    for completed in asyncio.as_completed(pending):
        index, output, is_correct = await completed
        correct += 1 if is_correct else 0
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        print(f"[{index}]", "🟢" if is_correct else "🔴", output, sep="\t")
    print(f"Total: {correct}/{len(tasks)}\tAcc: {100 * correct / len(tasks)}%")


async def query_decomposer(decomposer_agent, semaphore, index, question, gt):
    async with semaphore:
        is_correct = False
        try:
            lg_messages = {"messages": [{"role": "user", "content": question}]}
            state = await decomposer_agent.ainvoke(lg_messages)
            output = state["messages"][-1]["content"]
            is_correct = await verify_answer(question, output, gt)
        except Exception as error:
            output = f"ERROR: {error}"
        return index, output, is_correct


async def decomposer(
    tasks: list[tuple[str, str]],
    concurrency: int,
) -> None:
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
    decomposer_agent = create_decomposer_agent(
        decomposer_model=model,
        subagent_types=[
            {
                "subagent_type_id": "qwen3_6_35b_a3b_fp8_thinking",
                "description": "Qwen3.6-35B-A3B-FP8 with thinking enabled, without tools.",
                "assistant_id": "qwen3_6_35b_a3b_fp8_thinking",
                "url": "http://127.0.0.1:2024",
            }
        ],
    )
    semaphore = asyncio.Semaphore(concurrency)
    pending = [
        query_decomposer(decomposer_agent, semaphore, index, question, answer)
        for index, (question, answer) in enumerate(tasks, 1)
    ]
    correct = 0
    for completed in asyncio.as_completed(pending):
        index, output, is_correct = await completed
        correct += 1 if is_correct else 0
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        print(f"[{index}]", "🟢" if is_correct else "🔴", output, sep="\t")
    print(f"Total: {correct}/{len(tasks)}\tAcc: {100 * correct / len(tasks)}%")


async def main():
    parser = argparse.ArgumentParser(description="BrowseComp-Plus benchmark")
    parser.add_argument("mode", choices=["decomposer", "direct"])
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    tasks = load()
    if args.limit != -1:
        tasks = tasks[: args.limit]

    if args.mode == "decomposer":
        with timed("decomposer total"):
            await decomposer(tasks, args.concurrency)
    elif args.mode == "direct":
        with timed("direct total"):
            await direct(tasks, args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())
