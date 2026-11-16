"""Ordered all-answer correctness using MASBench's math-verify backend.

The multi-answer count/order protocol follows Salesforce/MAS-Orchestra at
0527c70a508eb49ed42d5dd49a91ea857551e242, rewards/utils/compute_score.py.
We do not import its training runtime or use its additional MathScorer/LLM judge.
"""
from math_verify import parse, verify
from math_verify.errors import TimeoutException


def boxed_answers(text):
    answers = []
    cursor = 0
    while (start := text.find("\\boxed{", cursor)) >= 0:
        cursor = start + 7
        depth = 1
        end = cursor
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth == 0:
            answers.append(text[cursor:end - 1].strip())
            cursor = end
    return answers


def evaluate(output, ground_truth):
    expected = [answer.strip() for answer in ground_truth.split("<<horizon>>")]
    predicted = boxed_answers(output)
    if len(expected) == 1:
        predicted = predicted[-1:] or [output.strip()]
    correct = []
    for prediction, gold in zip(predicted, expected):
        try:
            correct.append(bool(verify(parse(gold), parse(prediction))))
        except (Exception, TimeoutException):
            correct.append(False)
    passed = len(predicted) == len(expected) and all(correct)
    return {"passed": passed, "score": int(passed), "predicted": predicted,
            "expected": expected, "answer_correct": correct, "backend": "math-verify==0.8.0"}


def final_output(messages):
    if not messages:
        return ""
    last = messages[-1].get("data", messages[-1])
    if last.get("type") != "ai" or last.get("tool_calls"):
        return ""
    content = last.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content if block.get("type") == "text")
