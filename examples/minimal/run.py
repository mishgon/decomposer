import asyncio
import json
import logging
from pathlib import Path

from decomposer.agent_server import agent_server, invoke_and_capture
from decomposer.visualization import write_trace_html

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    config_path = Path(__file__).with_name("langgraph.json")
    async with agent_server(config_path) as url:
        final_state, error = await invoke_and_capture(
            url,
            "decomposer",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Create a beginner quiz with three sections: Python, SQL, "
                            "and machine learning. Each section must contain three "
                            "multiple-choice questions: one about a concept, one based "
                            "on a short code snippet or concrete example, and one about "
                            "a common mistake.\n\n"
                            "Each question must have exactly four answer choices, "
                            "one correct answer, and a one-sentence explanation. "
                            "Each section must contain fewer than 300 words. "
                            "Examples must be self-contained and require no external "
                            "resources. At the end, provide the complete quiz and "
                            "answer key. Write all agent assignments and the final "
                            "answer in English."
                        ),
                    }
                ]
            }
        )
    trace_path = Path(__file__).with_name("trace.json")
    trace_path.write_text(
        json.dumps(final_state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    html_path = trace_path.with_suffix(".html")
    rendered = write_trace_html(final_state, html_path)
    print(f"\nTrace saved to {trace_path}")
    if rendered:
        print(f"Visualization saved to {html_path}")
    if error is not None:
        raise error
    print(final_state["decomposer_agent_runs"][-1]["response"])


if __name__ == "__main__":
    asyncio.run(main())
