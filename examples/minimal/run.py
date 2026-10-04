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
                            "Составь тест для начинающих из трёх разделов: Python, SQL "
                            "и машинное обучение. В каждом разделе должно быть три вопроса "
                            "с вариантами ответа: один на понимание понятия, один по короткому "
                            "фрагменту кода или конкретному примеру и один о типичной ошибке.\n\n"
                            "У каждого вопроса должно быть ровно четыре варианта ответа, "
                            "один правильный ответ и объяснение в одном предложении. "
                            "Объём каждого раздела должен быть меньше 300 слов. "
                            "Примеры должны быть самодостаточными и не требовать внешних "
                            "ресурсов. В конце приведи полный тест и ключ с ответами. "
                            "Все задания агентам и итоговый ответ напиши по-русски."
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
    print(f"\nТрейс сохранён в {trace_path}")
    if rendered:
        print(f"Визуализация сохранена в {html_path}")
    if error is not None:
        raise error
    print(final_state["decomposer_agent_runs"][-1]["response"])


if __name__ == "__main__":
    asyncio.run(main())
