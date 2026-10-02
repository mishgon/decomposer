import asyncio
import logging
from pathlib import Path

from langchain_core.messages import convert_to_messages
from langgraph_sdk import get_client
from utils import agent_server

from render_messages import render_decomposer_messages

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    config_path = Path(__file__).with_name("langgraph.json")
    async with agent_server(config_path) as url:
        client = get_client(url=url)
        final_state = await client.runs.wait(
            None,
            "decomposer",
            input={
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
    messages = convert_to_messages(final_state["messages"])
    print(messages[-1].content)

    output_path = Path(__file__).with_name("messages.md")
    output_path.write_text(
        render_decomposer_messages(messages),
        encoding="utf-8",
    )
    print(f"\nSaved messages to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
