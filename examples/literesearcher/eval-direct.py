import asyncio

from langgraph_sdk import get_client

from lib import (
    ARTIFACTS_DIR,
    LITERESEARCHER_ASSISTANT_ID,
    LITERESEARCHER_URL,
    message_content_text,
    parse_eval_args,
    run_evaluation,
)

METHOD = "direct"
DEFAULT_OUTPUT_PATH = ARTIFACTS_DIR / "browsecomp_plus_eval_direct.json"
HISTORY_LIMIT = 1000
RECURSION_LIMIT = 200
TERMINAL_STATUSES = frozenset({"success", "error", "timeout", "interrupted"})


def create_direct_answerer(client):
    async def answer_question(question: str) -> str:
        thread = await client.threads.create()
        run = await client.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=LITERESEARCHER_ASSISTANT_ID,
            input={"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": RECURSION_LIMIT},
        )
        if run["status"] not in TERMINAL_STATUSES:
            await client.runs.join(
                thread_id=thread["thread_id"],
                run_id=run["run_id"],
            )
        completed_run = await client.runs.get(
            thread_id=thread["thread_id"],
            run_id=run["run_id"],
        )
        if completed_run["status"] != "success":
            error = completed_run.get("error")
            raise RuntimeError(
                f"LiteResearcher run ended with status "
                f"{completed_run['status']!r}: {error}"
            )

        history = await client.threads.get_history(
            thread_id=thread["thread_id"],
            limit=HISTORY_LIMIT,
            metadata={"run_id": run["run_id"]},
        )
        if not history:
            raise RuntimeError(f"No history found for run {run['run_id']!r}")
        if history[-1]["metadata"]["source"] != "input":
            raise RuntimeError("Run history was truncated")

        before_messages = history[-1]["values"]["messages"]
        after_messages = history[0]["values"]["messages"]
        run_messages = after_messages[len(before_messages) :]
        if not run_messages:
            raise RuntimeError(f"No messages found for run {run['run_id']!r}")

        last_message = run_messages[-1]
        if last_message["type"] != "ai" or last_message.get("tool_calls"):
            raise RuntimeError("LiteResearcher did not finish with a final answer")
        return message_content_text(last_message.get("content"))

    return answer_question


async def main() -> None:
    args = parse_eval_args(
        "Evaluate LiteResearcher directly on BrowseComp-Plus.",
        DEFAULT_OUTPUT_PATH,
    )
    client = get_client(
        url=LITERESEARCHER_URL,
        headers={"x-auth-scheme": "langsmith"},
    )
    await run_evaluation(
        answer_question=create_direct_answerer(client),
        method=METHOD,
        output_path=args.output,
        concurrency=args.concurrency,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    asyncio.run(main())
