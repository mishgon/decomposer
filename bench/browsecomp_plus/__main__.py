import time
import json
import asyncio
import argparse
from contextlib import contextmanager

from langgraph_sdk import get_client

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
            state = await client.runs.wait(
                None,
                "researcher",
                input={"messages": [{"role": "user", "content": question}]},
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


async def decomposer(
    tasks: list[tuple[str, str]],
    concurrency: int,
) -> None:
    raise NotImplementedError("decomposer mode is not implemented yet")


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
