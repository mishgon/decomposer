import asyncio
from pathlib import Path

from langchain_openrouter import ChatOpenRouter

from decomposer.core import create_decomposer_agent
from render_messages import render_decomposer_messages


async def main() -> None:
    decomposer_agent = create_decomposer_agent(
        decomposer_model=ChatOpenRouter(
            model="deepseek/deepseek-v4-flash-0731",
            temperature=1.0,
            top_p=1.0,
            reasoning={"effort": "high"},
        ),
        subagent_types=[
            {
                "subagent_type_id": "literesearcher_4b",
                "description": "researcher with thinking enabled and search tools.",
                "assistant_id": "literesearcher_4b",
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
                        "What recent movie produced quite a backlash "
                        "in social media since fans have been extensively "
                        "sharing TikToks of throwing popcorn after a particular phrase? "
                        "Find and quote full phrase "
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
