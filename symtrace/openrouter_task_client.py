from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SYSTEM_PROMPT = """You rewrite symbolic-tool trace tasks into concise, explicit user requests.

Return strict JSON with keys:
- task: a short explicit task statement a user could realistically ask
- answer_format: one short sentence describing the expected answer form

Rules:
- Do not say "transform ... to match reference output".
- Use imperative task wording like "Solve", "Compute", "Factor", "Construct", "Reduce".
- Keep the task under 20 words when possible.
- Be faithful to the given entry function, input, and final answer.
- Do not mention traces, SymPy internals, or tool calls.
"""


def _build_user_prompt(record: dict) -> str:
    return json.dumps(
        {
            "entry_function": record["task"]["entry_function"],
            "goal": record["task"].get("goal"),
            "input": record["task"]["input"],
            "final_answer": record.get("final_answer"),
            "tools": record.get("tools"),
        },
        sort_keys=True,
    )


def request_task_rewrite(
    record: dict,
    *,
    model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1/chat/completions",
    app_name: str | None = None,
    referer: str | None = None,
) -> dict:
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(record)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if app_name:
        headers["X-Title"] = app_name
    if referer:
        headers["HTTP-Referer"] = referer
    request = Request(base_url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    content = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return {
        "task": parsed["task"].strip(),
        "answer_format": parsed["answer_format"].strip(),
        "raw_response": body,
    }


def rewrite_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model: str,
    api_key: str,
    app_name: str | None = None,
    referer: str | None = None,
) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with input_file.open("r", encoding="utf-8") as src, output_file.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            rewrite = request_task_rewrite(
                record,
                model=model,
                api_key=api_key,
                app_name=app_name,
                referer=referer,
            )
            dst.write(
                json.dumps(
                    {
                        "episode_id": record["metadata"]["episode_id"],
                        "entry_function": record["task"]["entry_function"],
                        "task": rewrite["task"],
                        "answer_format": rewrite["answer_format"],
                        "final_answer": record.get("final_answer"),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrite trace tasks into explicit user-facing tasks via OpenRouter.")
    parser.add_argument("input_path", help="Input JSONL file, typically abstract_traces/train.jsonl.")
    parser.add_argument("output_path", help="Output JSONL file for rewritten tasks.")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="OpenRouter model identifier.")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY", help="Environment variable holding the API key.")
    parser.add_argument("--app-name", default="symtrace-task-rewriter", help="Optional OpenRouter app title.")
    parser.add_argument("--referer", default=None, help="Optional HTTP referer header.")
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Missing API key in ${args.api_key_env}", file=sys.stderr)
        return 2

    rewrite_file(
        args.input_path,
        args.output_path,
        model=args.model,
        api_key=api_key,
        app_name=args.app_name,
        referer=args.referer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
