"""Pinned upstream item-F1; failed judge requests never become a model score."""
import ast
import asyncio
import json
import os
import re
import time
from uuid import uuid4

from gyms.wideseek.runtime import model, save
from gyms.wideseek.vendor.table_reward import evaluate_markdown, extract_final_answer
from gyms.wideseek.vendor.qa_prompt import LLM_JUDGE_PROMPT


def validate_judge(text, messages):
    if messages[0]["content"].startswith("You are an evaluation assistant."):
        if text.strip().lower() not in {"correct", "incorrect"}:
            raise ValueError("QA judge must return Correct or Incorrect")
        return text
    matches = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not matches:
        raise ValueError("Judge omitted fenced JSON")
    result = json.loads(matches[-1])
    if not isinstance(result, dict):
        raise ValueError("Judge response must be an object")
    if messages[0]["content"].startswith("You are an expert in grading"):
        content = messages[1]["content"]
        pairs = ast.literal_eval(content[content.index("{"):].strip())
        if set(result) != set(pairs) or any(type(v) not in (int, float) or v not in (0, 1) for v in result.values()):
            raise ValueError("Judge must score every cell exactly once with 0 or 1")
    elif any(not isinstance(v, str) for v in result.values()):
        raise ValueError("Judge alignment values must be strings")
    return text


async def evaluate(task, answer, path, *, judge_model_id=None):
    is_table = bool(task["unique_columns"])
    metric = "item_f1" if is_table else "qa_accuracy"
    if is_table:
        reference = extract_final_answer(task["answer"], mode="markdown", strict=False)
        if reference is None or reference.empty:
            raise ValueError("Invalid dataset reference table")
    parsed = extract_final_answer(answer, mode="markdown" if is_table else "boxed", strict=True)
    if parsed is None or (is_table and parsed.empty):
        return {"status": "scored", "metric": metric, "score": 0., "format_ok": False}
    judge_model_id = judge_model_id or os.environ.get("WS_JUDGE_MODEL") or os.environ.get("WS_MODEL", "Qwen/Qwen3.5-4B")
    judge_model = model(judge_model_id)
    semaphore = asyncio.Semaphore(2)
    errors = []

    async def judge(messages):
        row = {"started_at": time.time(), "messages": messages, "model": judge_model_id,
               "temperature": 0., "presence_penalty": 0., "thinking": False}
        try:
            async with semaphore:
                response = await judge_model.ainvoke(messages, temperature=0., presence_penalty=0.)
            row["response"] = response.model_dump(mode="json")
            return validate_judge(response.content, messages)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(row["error"])
            # Upstream swallows judge errors. Keep them externally and invalidate the score.
            return "```json\n{}\n```" if is_table else "Incorrect"
        finally:
            row["finished_at"] = time.time()
            await asyncio.to_thread(save, path / "judge_calls" / f"{uuid4().hex}.json", row)

    try:
        if is_table:
            label = dict(task)
            evaluation = task.get("metadata", {}).get("evaluation")
            if isinstance(evaluation, str):
                evaluation = json.loads(evaluation)
            if isinstance(evaluation, dict):
                label["required"] = evaluation.get("required", [])
            score, format_ok = await evaluate_markdown(parsed, label, judge, False)
        else:
            reply = await judge([
                {"role": "system", "content": "You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer."},
                {"role": "user", "content": LLM_JUDGE_PROMPT.format(question=task["question"],
                    correct_answer=task["answer"], response=parsed)}])
            # Same native QA decision; malformed/API responses are invalidated below.
            score = float("correct" in reply.strip().lower() and "incorrect" not in reply.strip().lower())
            format_ok = True
    finally:
        await judge_model.http_async_client.aclose()
    return ({"status": "judge_error", "metric": metric, "score": None, "errors": errors} if errors else
            {"status": "scored", "metric": metric, "score": float(score), "format_ok": bool(format_ok)})
