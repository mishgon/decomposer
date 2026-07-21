import asyncio
from pathlib import Path

from langchain_openrouter import ChatOpenRouter

from decomposer.core import create_decomposer_agent
from render_messages import render_decomposer_messages


async def main() -> None:
    decomposer_agent = create_decomposer_agent(
        decomposer_model=ChatOpenRouter(
            model="z-ai/glm-5.2",
            temperature=1.0,
            top_p=0.95,
            max_tokens=131072,
            reasoning={"effort": "high"},
        ),
        subagent_types=[
            {
                "subagent_type_id": "qwen3_6_35b_a3b_fp8_thinking",
                "description": "Qwen3.6-35B-A3B-FP8 with thinking enabled, without tools.",
                "assistant_id": "qwen3_6_35b_a3b_fp8_thinking",
                "url": "http://127.0.0.1:2024",
            }
        ],
    )
    final_state = await decomposer_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "An online service handles 10,000,000 requests per month and "
                        "expects traffic to grow by 40%. It must choose the lowest-cost "
                        "hosting plan that meets all requirements both now and after "
                        "the growth: availability of at least 99.95%, p95 latency of at "
                        "most 150 ms, and monthly cost of at most $5,200. Plan A has a "
                        "$2,500 fixed monthly cost, costs $0.12 per 1,000 requests, has "
                        "99.97% availability, and 110 ms p95 latency. Plan B has a "
                        "$1,500 fixed monthly cost, costs $0.25 per 1,000 requests, has "
                        "99.99% availability, and 135 ms p95 latency. Plan C has a "
                        "$1,000 fixed monthly cost, costs $0.08 per 1,000 requests, has "
                        "99.90% availability, and 90 ms p95 latency. Calculate the "
                        "current and forecast monthly costs for every plan, assess every "
                        "requirement, and recommend a plan. For the recommended plan, "
                        "also determine the largest increase in its per-1,000-request "
                        "price it could absorb at forecast traffic before either exceeding "
                        "the budget or becoming more expensive than another plan that "
                        "meets all requirements. Identify which limit binds. Show the "
                        "calculations and summarize the result in a concise table."
                    ),
                }
            ]
        }
    )
    print(final_state["messages"][-1].content)

    output_path = Path(__file__).with_name("messages.md")
    output_path.write_text(
        render_decomposer_messages(final_state["messages"]),
        encoding="utf-8",
    )
    print(f"\nSaved messages to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
