import ast

import pytest

from prompts import DECOMPOSER_EXAMPLES
from prototype import (
    EXEC_BUILTINS,
    exec_func,
    extract_func_body,
    render_func_src,
    validate_func_body,
    validate_func_src,
)


FUNC_SPEC = "def main(context, context_qa_model):"


def render_and_parse(func_body: str) -> str:
    func_src = render_func_src(FUNC_SPEC, func_body)
    validate_func_src(func_src)
    ast.parse(func_src)
    return func_src


FUNC_BODY_CASES = [
    pytest.param(
        "answer_1 = context_qa_model('Q1', context)\n"
        "answer_2 = context_qa_model(f'Q2 {answer_1}', context)\n"
        "return answer_2",
        "answer_1 = context_qa_model('Q1', context)\n"
        "answer_2 = context_qa_model(f'Q2 {answer_1}', context)\n"
        "return answer_2",
        id="unindented",
    ),
    pytest.param(
        "    answer_1 = context_qa_model('Q1', context)\n"
        "    answer_2 = context_qa_model(f'Q2 {answer_1}', context)\n"
        "    return answer_2",
        "answer_1 = context_qa_model('Q1', context)\n"
        "answer_2 = context_qa_model(f'Q2 {answer_1}', context)\n"
        "return answer_2",
        id="uniformly-indented",
    ),
    pytest.param(
        "\n\n    answer_1 = context_qa_model('Q1', context)\n"
        "    return answer_1\n\n",
        "answer_1 = context_qa_model('Q1', context)\n"
        "return answer_1",
        id="surrounding-blank-lines",
    ),
    pytest.param(
        "    answer_1 = context_qa_model('Q1', context)\n"
        "\n"
        "    answer_2 = context_qa_model(f'Q2 {answer_1}', context)\n"
        "    return answer_2",
        "answer_1 = context_qa_model('Q1', context)\n"
        "\n"
        "answer_2 = context_qa_model(f'Q2 {answer_1}', context)\n"
        "return answer_2",
        id="internal-blank-line",
    ),
    pytest.param(
        "answer_1 = context_qa_model('Q1', context)\n"
        "if answer_1:\n"
        "    answer_2 = context_qa_model(f'Q2 {answer_1}', context)\n"
        "    return answer_2\n"
        "return 'unknown'",
        "answer_1 = context_qa_model('Q1', context)\n"
        "if answer_1:\n"
        "    answer_2 = context_qa_model(f'Q2 {answer_1}', context)\n"
        "    return answer_2\n"
        "return 'unknown'",
        id="nested",
    ),
    pytest.param(
        "    answer_1 = context_qa_model('Q1', context)\n"
        "    if answer_1:\n"
        "        answer_2 = context_qa_model(f'Q2 {answer_1}', context)\n"
        "        return answer_2\n"
        "    return 'unknown'",
        "answer_1 = context_qa_model('Q1', context)\n"
        "if answer_1:\n"
        "    answer_2 = context_qa_model(f'Q2 {answer_1}', context)\n"
        "    return answer_2\n"
        "return 'unknown'",
        id="uniformly-indented-nested",
    ),
    pytest.param(
        "answer_1 = context_qa_model('Q1', context)\n"
        "    answer_2 = context_qa_model(f'Q2 {answer_1}', context)\n"
        "    return answer_2",
        "answer_1 = context_qa_model('Q1', context)\n"
        "answer_2 = context_qa_model(f'Q2 {answer_1}', context)\n"
        "return answer_2",
        id="missing-first-line-indent",
    ),
    pytest.param(
        "answer_1 = context_qa_model('Q1', context)\n"
        "    if answer_1:\n"
        "        answer_2 = context_qa_model(f'Q2 {answer_1}', context)\n"
        "        return answer_2\n"
        "    return 'unknown'",
        "answer_1 = context_qa_model('Q1', context)\n"
        "if answer_1:\n"
        "    answer_2 = context_qa_model(f'Q2 {answer_1}', context)\n"
        "    return answer_2\n"
        "return 'unknown'",
        id="missing-first-line-indent-with-nesting",
    ),
    pytest.param(
        "if answer_1:\n"
        "        return answer_1",
        "if answer_1:\n"
        "        return answer_1",
        id="parseable-overindented-nested-block",
    ),
    pytest.param(
        "reasoning text</think>\n"
        "    answer_1 = context_qa_model('Q1', context)\n"
        "    return answer_1",
        "answer_1 = context_qa_model('Q1', context)\n"
        "return answer_1",
        id="qwen-template-thinking-prefix",
    ),
    pytest.param(
        "<think>reasoning text</think>\n"
        "    answer_1 = context_qa_model('Q1', context)\n"
        "    return answer_1",
        "answer_1 = context_qa_model('Q1', context)\n"
        "return answer_1",
        id="complete-thinking-block",
    ),
]


@pytest.mark.parametrize(("raw_output", "expected_body"), FUNC_BODY_CASES)
def test_extract_func_body(raw_output: str, expected_body: str) -> None:
    func_body = extract_func_body(raw_output)
    assert func_body == expected_body
    render_and_parse(func_body)


def test_extract_func_body_returns_none_for_unclosed_thinking() -> None:
    assert extract_func_body("<think>still thinking") is None


@pytest.mark.parametrize("example", DECOMPOSER_EXAMPLES)
def test_decomposer_few_shot_examples_use_generation_boundary_format(
    example: dict,
) -> None:
    lines = example["func_body"].splitlines()
    assert not lines[0].startswith((" ", "\t"))
    for line in lines[1:]:
        if line.strip():
            assert line.startswith("    ")
    func_body = extract_func_body(example["func_body"])
    assert not func_body.startswith(" ")
    render_and_parse(func_body)


@pytest.mark.parametrize(
    ("raw_output", "error"),
    [
        pytest.param(
            "```python\n"
            "answer_1 = context_qa_model('Q1', context)\n"
            "return answer_1\n"
            "```",
            "Markdown fences",
            id="markdown-fence",
        ),
        pytest.param(
            "answer_1 = context_qa_model('Q1', context)\n"
            "\treturn answer_1",
            "spaces, not tabs",
            id="tab-indent",
        ),
        pytest.param("\n\n", "empty", id="empty"),
    ],
)
def test_extract_func_body_preserves_invalid_output_for_validation_errors(
    raw_output: str,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        validate_func_body(extract_func_body(raw_output))


def test_render_adds_one_outer_indent() -> None:
    func_src = render_func_src(
        FUNC_SPEC,
        "answer_1 = context_qa_model('Q1', context)\n"
        "return answer_1",
    )
    assert (
        func_src
        == "def main(context, context_qa_model):\n"
        "    answer_1 = context_qa_model('Q1', context)\n"
        "    return answer_1"
    )


def test_render_preserves_nested_relative_indent() -> None:
    func_src = render_func_src(
        FUNC_SPEC,
        "answer_1 = context_qa_model('Q1', context)\n"
        "if answer_1:\n"
        "    return answer_1\n"
        "return 'unknown'",
    )
    assert (
        func_src
        == "def main(context, context_qa_model):\n"
        "    answer_1 = context_qa_model('Q1', context)\n"
        "    if answer_1:\n"
        "        return answer_1\n"
        "    return 'unknown'"
    )
    validate_func_src(func_src)


def test_validate_func_body_rejects_first_line_indentation() -> None:
    with pytest.raises(ValueError, match="first generated function body line"):
        validate_func_body("    answer_1 = 1\nreturn answer_1")


@pytest.mark.parametrize(
    ("func_body", "error"),
    [
        pytest.param("import os\nreturn 'x'", "import modules", id="import"),
        pytest.param("return eval('1 + 1')", "eval", id="forbidden-name"),
        pytest.param("return context.__class__", "dunder", id="dunder-attribute"),
    ],
)
def test_validate_func_src_rejects_forbidden_code(
    func_body: str,
    error: str,
) -> None:
    func_src = render_func_src(FUNC_SPEC, func_body)
    with pytest.raises(ValueError, match=error):
        validate_func_src(func_src)


def test_generated_body_can_raise_value_error() -> None:
    func_src = render_func_src(FUNC_SPEC, 'raise ValueError("ambiguous question")')
    validate_func_src(func_src)
    namespace = {"__builtins__": EXEC_BUILTINS.copy()}
    exec(func_src, namespace)

    with pytest.raises(ValueError, match="ambiguous question"):
        namespace["main"]("", lambda question, context: "")


def test_exec_func_returns_execution_error_for_generated_runtime_error() -> None:
    func_src = render_func_src(FUNC_SPEC, 'raise ValueError("ambiguous question")')

    prediction, tool_calling_logs, execution_error = exec_func(
        func_src,
        context="",
        qa_model="unused",
    )

    assert prediction is None
    assert tool_calling_logs == []
    assert execution_error == "ValueError: ambiguous question"


def test_exec_func_returns_prediction_without_execution_error() -> None:
    func_src = render_func_src(FUNC_SPEC, "return 42")

    prediction, tool_calling_logs, execution_error = exec_func(
        func_src,
        context="",
        qa_model="unused",
    )

    assert prediction == "42"
    assert tool_calling_logs == []
    assert execution_error is None


def test_render_func_src_rejects_none_body() -> None:
    with pytest.raises(ValueError, match="missing"):
        render_func_src(FUNC_SPEC, None)
