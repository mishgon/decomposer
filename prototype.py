"""Minimal static Decomposer prototype for MuSiQue.

This first slice loads one MuSiQue example, asks Decomposer for a function
body, and executes the resulting function with a context QA tool.
"""

import argparse
import ast
import json
import os
import textwrap
from pathlib import Path

import requests

from prompts import (
    DECOMPOSER_FEW_SHOT_PROMPT,
    build_decomposer_messages,
    build_func_spec,
    build_context_qa_messages,
)
from sampling import build_sampling_params


DEFAULT_DATA_PATH = Path("data/musique_ans_v1.0_dev.jsonl")
DEFAULT_DECOMPOSER_MODEL = "qwen/qwen3.6-27b"
DECOMPOSER_ENABLE_THINKING = True
DEFAULT_QA_MODEL = "qwen/qwen3.6-27b"
QA_ENABLE_THINKING = True
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
EXEC_BUILTINS = {
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "ValueError": ValueError,
}


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


def call_openrouter(
    messages: list[dict],
    model: str,
    sampling_params: dict,
    enable_thinking: bool | None = None,
    reasoning: dict | None = None,
    verbose: bool = False,
) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    payload = {"model": model, "messages": messages}
    payload.update(sampling_params)
    if reasoning is not None:
        payload["reasoning"] = reasoning
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    try:
        if verbose:
            print(
                f"Calling OpenRouter: model={model}, enable_thinking={enable_thinking}",
                flush=True,
            )
        response = requests.post(
            OPENROUTER_URL,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )
        if verbose:
            print(f"OpenRouter response: HTTP {response.status_code}", flush=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        response_body = exc.response.text if exc.response is not None else ""
        raise RuntimeError(
            f"OpenRouter request failed for model {model}: "
            f"{exc}: {response_body}"
        ) from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"OpenRouter returned a non-JSON response for model {model}: "
            f"{response.text}"
        ) from exc

    try:
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("message.content is not a string")
        return content
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"OpenRouter returned an unexpected response for model {model}: "
            f"{json.dumps(data, ensure_ascii=False)}"
        ) from exc


def _extract_response_without_thinking(raw_output: str) -> str | None:
    # Keep spaces intact because generated code may rely on indentation.
    # Only surrounding blank lines are formatting noise here.
    text = raw_output.strip("\n")
    lowered = text.lower()
    # Qwen chat templates may put the opening tag in the prompt, leaving only
    # the generated closing tag in the returned text.
    if "</think>" in lowered:
        return text[lowered.rfind("</think>") + len("</think>") :].strip("\n")
    if "<think>" in lowered:
        return None
    return text


def extract_answer(raw_output: str) -> str | None:
    output = _extract_response_without_thinking(raw_output)
    if output is None:
        return None
    return output.strip()


def is_unanswerable_answer(answer: str) -> bool:
    return answer.strip().rstrip(".").casefold() == "unanswerable"


def _dedent_func_body(text: str) -> str:
    return textwrap.dedent(text).strip("\n")


def _indent_func_body(func_body: str) -> str:
    return "\n".join(
        f"    {line}" if line.strip() else line
        for line in func_body.splitlines()
    )


def _render_body_for_parse(func_body: str) -> str:
    return f"def main():\n{_indent_func_body(func_body)}"


def _is_parseable_func_body(func_body: str) -> bool:
    try:
        ast.parse(_render_body_for_parse(func_body))
    except SyntaxError:
        return False
    return True


def _with_first_line_indented(func_body: str) -> str:
    lines = func_body.splitlines()
    for index, line in enumerate(lines):
        if line.strip():
            lines[index] = f"    {line}"
            break
    return "\n".join(lines)


def extract_func_body(raw_output: str) -> str | None:
    output = _extract_response_without_thinking(raw_output)
    if output is None:
        return None

    func_body = _dedent_func_body(output)
    repaired_body = _dedent_func_body(_with_first_line_indented(func_body))
    candidates = [func_body]
    if repaired_body != func_body:
        candidates.append(repaired_body)

    for candidate in candidates:
        if _is_parseable_func_body(candidate):
            return candidate
    return func_body


def validate_func_body(func_body: str | None) -> None:
    if func_body is None:
        raise ValueError("Generated function body is missing.")
    if "```" in func_body:
        raise ValueError("Generated function body must not contain Markdown fences.")
    lines = func_body.strip("\n").splitlines()
    nonempty_lines = [line for line in lines if line.strip()]
    if not nonempty_lines:
        raise ValueError("Generated function body is empty.")
    if nonempty_lines[0].startswith((" ", "\t")):
        raise ValueError("The first generated function body line must be unindented.")
    for line in nonempty_lines:
        if line.startswith("\t"):
            raise ValueError("Generated function body must use spaces, not tabs.")


def render_func_src(func_spec: str, func_body: str | None) -> str:
    validate_func_body(func_body)
    func_body = func_body.strip("\n")
    return f"{func_spec}\n{_indent_func_body(func_body)}"


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


def format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def exec_func(
    func_src: str,
    context: str,
    qa_model: str,
    qa_enable_thinking: bool = QA_ENABLE_THINKING,
    verbose: bool = False,
) -> tuple[str | None, list[dict], str | None]:
    validate_func_src(func_src)
    namespace = {"__builtins__": EXEC_BUILTINS.copy()}
    exec(func_src, namespace)
    tool_calling_logs = []

    # tools
    def context_qa_model(question: str, context: str) -> str:
        if verbose:
            print(f"\nContext QA: {question}", flush=True)
        messages = build_context_qa_messages(question, context)
        raw_output = call_openrouter(
            messages,
            qa_model,
            build_sampling_params(qa_model, qa_enable_thinking, task="general"),
            enable_thinking=qa_enable_thinking,
            reasoning=(
                {"max_tokens": 2048, "exclude": True}
                if qa_enable_thinking
                else {"effort": "none"}
            ),
            verbose=verbose,
        )
        answer = extract_answer(raw_output)
        tool_calling_logs.append(
            {"tool": "context_qa_model", "question": question, "answer": answer}
        )
        if not answer:
            raise ValueError("Context QA output did not contain an answer.")
        if is_unanswerable_answer(answer):
            raise ValueError("Context QA question is unanswerable.")
        return answer

    try:
        answer = namespace["main"](context, context_qa_model)
    except Exception as exc:
        return None, tool_calling_logs, format_error(exc)
    if answer is not None:
        answer = str(answer)
    return answer, tool_calling_logs, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--decomposer-model", default=DEFAULT_DECOMPOSER_MODEL)
    parser.add_argument("--qa-model", default=DEFAULT_QA_MODEL)
    parser.add_argument(
        "--qa-thinking",
        action=argparse.BooleanOptionalAction,
        default=QA_ENABLE_THINKING,
    )
    parser.add_argument("--show-examples", action="store_true")
    parser.add_argument(
        "--decomposer-thinking",
        action=argparse.BooleanOptionalAction,
        default=DECOMPOSER_ENABLE_THINKING,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.data)
    example = examples[args.index]

    print(f"ID: {example['id']}")
    print(f"Question: {example['question']}")
    print(f"Answer: {example['answer']}")
    print("\nInput function spec:")
    print(build_func_spec(example["question"]))

    if args.show_examples:
        print("\nDecomposition examples:")
        print(DECOMPOSER_FEW_SHOT_PROMPT)
        return

    context = format_context(example)
    func_spec = build_func_spec(example["question"])
    messages = build_decomposer_messages(func_spec)
    raw_output = call_openrouter(
        messages,
        args.decomposer_model,
        build_sampling_params(
            args.decomposer_model,
            args.decomposer_thinking,
            task="code",
        ),
        enable_thinking=args.decomposer_thinking,
        reasoning=(
            {"max_tokens": 2048, "exclude": True}
            if args.decomposer_thinking
            else {"effort": "none"}
        ),
        verbose=args.verbose,
    )
    func_body = extract_func_body(raw_output)
    print("\nGenerated function body:")
    print(func_body)

    func_src = render_func_src(func_spec, func_body)
    print("\nRendered function source:")
    print(func_src)

    prediction, tool_calling_logs, execution_error = exec_func(
        func_src,
        context,
        args.qa_model,
        qa_enable_thinking=args.qa_thinking,
        verbose=args.verbose,
    )
    print("\nTool calling logs:")
    for call in tool_calling_logs:
        print(f"- {call['question']} -> {call['answer']}")
    print("\nExecution error:")
    print(execution_error)
    print("\nPrediction:")
    print(prediction)
    print("\nGold answer:")
    print(example["answer"])


if __name__ == "__main__":
    main()
