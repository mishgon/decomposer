"""Answer-equivalence scoring. Judge failures remain explicitly unscored."""
import json
import re

from langchain_core.messages import message_to_dict
from decomposer.models import create_model
from gyms.browsecomp.agents import JUDGE_MODEL

FINAL_FORMAT = '\nEnd with a fenced JSON object containing "final_answer" (a short answer) and "gold_sources" (supporting URLs).'


def extract_answer(output):
    for block in reversed(re.findall(r"```json\s*(.*?)\s*```", output, re.DOTALL | re.IGNORECASE)):
        try:
            value = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("final_answer"), str):
            return value["final_answer"].strip()
    return ""


async def evaluate(task, output, directory):
    answer = extract_answer(output)
    if not answer:
        return {"status": "scored", "score": 0, "passed": False, "reason": "missing_final_answer"}
    if answer.casefold() == task["answer"].strip().casefold():
        return {"status": "scored", "score": 1, "passed": True, "reason": "exact_match"}
    messages = [
        {"role": "system", "content": 'Judge answer equivalence using the question. Accept equivalent entities, quantities and aliases. Reject different or ambiguous answers. Return only JSON: {"match": true or false}.'},
        {"role": "user", "content": json.dumps({"question": task["question"],
                                                 "gold": task["answer"], "generated": answer})},
    ]
    record = {"model": JUDGE_MODEL, "messages": messages}
    try:
        response = await create_model(JUDGE_MODEL).bind(temperature=0).ainvoke(messages)
        record["response"] = message_to_dict(response)
        content = response.content.strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
        verdict = json.loads(content)["match"]
        if not isinstance(verdict, bool):
            raise ValueError("Judge match must be a boolean")
        return {"status": "scored", "score": int(verdict), "passed": verdict, "reason": "judge"}
    except Exception as error:
        record["error"] = repr(error)
        return {"status": "evaluation_error", "score": None, "passed": False, "error": repr(error)}
    finally:
        (directory / "judge.json").write_text(json.dumps(record, indent=2))
