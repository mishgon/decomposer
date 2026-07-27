import asyncio

from langchain_openrouter import ChatOpenRouter

from decomposer.core import create_decomposer_agent
from lib import (
    ARTIFACTS_DIR,
    LITERESEARCHER_ASSISTANT_ID,
    LITERESEARCHER_URL,
    message_text,
    parse_eval_args,
    run_evaluation,
)

DECOMPOSER_MODEL_NAME = "z-ai/glm-5.2"
METHOD = "decomposer"
DEFAULT_OUTPUT_PATH = ARTIFACTS_DIR / "browsecomp_plus_eval.jsonl"


def build_decomposer_agent():
    return create_decomposer_agent(
        decomposer_model=ChatOpenRouter(
            model=DECOMPOSER_MODEL_NAME,
            temperature=1.0,
            top_p=0.95,
            max_tokens=131072,
            reasoning={"effort": "high"},
        ),
        subagent_types=[
            {
                "subagent_type_id": LITERESEARCHER_ASSISTANT_ID,
                "description": "researcher with thinking enabled and search tools.",
                "assistant_id": LITERESEARCHER_ASSISTANT_ID,
                "url": LITERESEARCHER_URL,
            }
        ],
    )


async def main() -> None:
    args = parse_eval_args(
        "Evaluate Decomposer with LiteResearcher on BrowseComp-Plus.",
        DEFAULT_OUTPUT_PATH,
    )
    decomposer_agent = build_decomposer_agent()

    async def answer_question(question: str) -> str:
        final_state = await decomposer_agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
	    config={"recursion_limit": 45},
        )
        return message_text(final_state["messages"][-1])

    await run_evaluation(
        answer_question=answer_question,
        method=METHOD,
        output_path=args.output,
        concurrency=args.concurrency,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    asyncio.run(main())
