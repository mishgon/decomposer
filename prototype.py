"""Minimal static Decomposer prototype for MuSiQue.

This first slice loads one MuSiQue example, asks Decomposer for a function
body, and executes the resulting function with a single-hop QA tool.
"""

import argparse
import ast
import json
import os
from pathlib import Path
from urllib import error, request

from prompts import (
    DECOMPOSER_FEW_SHOT_PROMPT,
    build_decomposer_messages,
    build_single_hop_qa_messages,
)


DEFAULT_DATA_PATH = Path("data/musique_ans_v1.0_dev.jsonl")
DEFAULT_DECOMPOSER_MODEL = "qwen/qwen3.6-27b"
DEFAULT_QA_MODEL = "qwen/qwen3.5-9b"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_examples(path: Path) -> list[dict]:
    examples = []
    with path.open() as file:
        for line in file:
            example = json.loads(line)
            if example["answerable"]:
                examples.append(example)
    return examples


def format_context(example: dict) -> str:
    paragraphs = []
    for paragraph in example["paragraphs"]:
        paragraphs.append(
            f"[{paragraph['idx']}] {paragraph['title']}\n"
            f"{paragraph['paragraph_text']}"
        )
    return "\n\n".join(paragraphs)


def build_func_spec(example: dict) -> str:
    return f'''def main(context, answer_single_hop_question):
    """Answer the question: {example["question"]}

    Args:
        context (str): A context that contains the necessary information to
            answer a question.
        answer_single_hop_question (Callable[[str, str], str]): Function that
            takes two arguments:
            - question (str): A single-hop question answerable from context.
            - context (str): The same formatted MuSiQue context string passed
              to main.
            It returns a short answer string grounded in the context.

    Returns:
        str: The final answer to the question.
    """'''


def call_openrouter(
    messages: list[dict],
    model: str,
    enable_thinking: bool | None = None,
) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    payload = json.dumps(payload).encode()
    http_request = request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=120) as response:
            data = json.loads(response.read().decode())
    except error.HTTPError as exc:
        response_body = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"OpenRouter request failed for model {model}: "
            f"HTTP {exc.code} {exc.reason}: {response_body}"
        ) from exc
    return data["choices"][0]["message"]["content"].strip("\n")


def validate_func_body(func_body: str) -> None:
    lines = func_body.strip("\n").splitlines()
    nonempty_lines = [line for line in lines if line.strip()]
    if not nonempty_lines:
        raise ValueError("Generated function body is empty.")
    if nonempty_lines[0].startswith((" ", "\t")):
        raise ValueError("The first generated function body line must be unindented.")
    for line in nonempty_lines[1:]:
        if not line.startswith("    "):
            raise ValueError("Generated function body lines after the first must be indented.")


def render_func_src(func_spec: str, func_body: str) -> str:
    func_body = func_body.strip("\n")
    validate_func_body(func_body)
    return f"{func_spec}\n    {func_body}"


def validate_func_src(func_src: str) -> None:
    tree = ast.parse(func_src)
    forbidden_names = {"eval", "exec", "open", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Generated body must not import modules.")
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            raise ValueError(f"Generated body must not use {node.id}.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Generated body must not access dunder attributes.")


def exec_func(func_src: str, context: str, qa_model: str) -> tuple[str, list[dict]]:
    validate_func_src(func_src)
    namespace = {"__builtins__": {"len": len, "str": str, "int": int, "float": float}}
    exec(func_src, namespace)
    tool_calling_logs = []

    # tools
    def answer_single_hop_question(question: str, context: str) -> str:
        messages = build_single_hop_qa_messages(question, context)
        answer = call_openrouter(messages, qa_model, enable_thinking=False)
        tool_calling_logs.append({"tool": "answer_single_hop_question", "question": question, "answer": answer})
        return answer

    answer = namespace["main"](context, answer_single_hop_question)
    return answer, tool_calling_logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--decomposer-model", default=DEFAULT_DECOMPOSER_MODEL)
    parser.add_argument("--qa-model", default=DEFAULT_QA_MODEL)
    parser.add_argument("--show-examples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.data)
    example = examples[args.index]

    print(f"ID: {example['id']}")
    print(f"Question: {example['question']}")
    print(f"Answer: {example['answer']}")
    print("\nInput function spec:")
    print(build_func_spec(example))

    if args.show_examples:
        print("\nDecomposition examples:")
        print(DECOMPOSER_FEW_SHOT_PROMPT)
        return

    func_spec = build_func_spec(example)
    messages = build_decomposer_messages(func_spec)
    func_body = call_openrouter(messages, args.decomposer_model, enable_thinking=True)
    print("\nGenerated function body:")
    print(func_body)

    func_src = render_func_src(func_spec, func_body)
    print("\nRendered function source:")
    print(func_src)

    prediction, tool_calling_logs = exec_func(func_src, format_context(example), args.qa_model)
    print("\nTool calling logs:")
    for call in tool_calling_logs:
        print(f"- {call['question']} -> {call['answer']}")
    print("\nPrediction:")
    print(prediction)
    print("\nGold answer:")
    print(example["answer"])


if __name__ == "__main__":
    main()
