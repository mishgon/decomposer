import asyncio
import logging
from pathlib import Path

from decomposer.models import create_model

from decomposer.core import create_decomposer_agent
from render_messages import render_decomposer_messages

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    decomposer_agent = create_decomposer_agent(
        decomposer_model=create_model("qwen_3_8_flash_next_non_thinking"),
        subagent_types=[
            {
                "subagent_type_id": "qwen_3_5_4b_unlooped_thinking",
                "description": "Qwen3.5-4B unlooped with thinking enabled, without tools.",
                "assistant_id": "qwen_3_5_4b_unlooped_thinking",
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
                        "Create a beginner quiz with three sections: Python, SQL, and "
                        "machine learning. Each section must contain three multiple-choice "
                        "questions: one conceptual question, one question about a short "
                        "code snippet or concrete example, and one question about a "
                        "common mistake.\n\n"
                        "Each question must have exactly four options, one correct "
                        "answer, and a one-sentence explanation. Keep each section "
                        "under 300 words. Use self-contained examples, require no "
                        "external resources, and finish with the complete quiz and "
                        "answer key."
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
