import json
import os
import re
from functools import lru_cache

from openai import AsyncOpenAI

_URL_RE = re.compile(r"https?://\S+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_FENCED_JSON_RE = re.compile(r"```json\s*([\{\[][\s\S]*?[\}\]])\s*```", re.IGNORECASE)
_VERDICT_RE = re.compile(r'"match"\s*:\s*(true|false)', re.IGNORECASE)

_JUDGE_PROMPT = """You are an answer-equivalence judge.

You will receive:
- a QUESTION
- a GOLD reference answer
- a GENERATED answer produced by an agent

Decide whether the GENERATED answer is correct relative to the GOLD. Reply with a SINGLE fenced JSON block and nothing else:

```json
{"match": true}
```

or

```json
{"match": false}
```

Equivalence criteria (mark match=true when ANY hold):
- Both identify the SAME named entity (full name, acronym, or common alias; e.g. "United Kingdom" ~ "UK" ~ "Britain").
- Both identify the SAME number or quantity, even if units/phrasing differ (e.g. "32" ~ "32 teams" ~ "thirty-two").
- Both identify the SAME concept, even worded differently (e.g. "passport-free travel" ~ "Schengen area" when the question asks what zone provides that benefit and gold says one, gen says the other).
- One is a substring of the other after lowercasing/stripping punctuation.

Non-equivalence (match=false):
- Different entities, numbers, or concepts.
- Generated answer is empty, a refusal, or off-topic.
- Generated answer is strictly more specific OR more general than gold in a way that changes correctness (e.g. gold="Italy" gen="Rome, Italy" is fine; gold="European Union" gen="Germany" is NOT a match even though Germany is in the EU).

Ignore: citation URLs, trailing punctuation, capitalization, articles ("the", "a"), whitespace. The QUESTION is your disambiguator — if the generated answer plausibly answers the question with the same fact as gold, mark match=true."""


def _normalize(value: str) -> set[str]:
    value = _URL_RE.sub("", value)
    value = _PUNCT_RE.sub(" ", value)
    value = _WS_RE.sub(" ", value).strip().lower()
    return set(value.split())


def _inclusion_match(gold: str, generated: str) -> bool:
    gold_tokens = _normalize(gold)
    generated_tokens = _normalize(generated)
    return bool(
        gold_tokens
        and generated_tokens
        and (gold_tokens <= generated_tokens or generated_tokens <= gold_tokens)
    )


def _extract_final_answer(output: str) -> str:
    matches = _FENCED_JSON_RE.findall(output)
    if not matches:
        return ""
    try:
        action = json.loads(matches[-1])
    except json.JSONDecodeError:
        return ""
    actions = action if isinstance(action, list) else [action]
    final = next(
        (
            item["final_answer"]
            for item in actions
            if isinstance(item, dict) and "final_answer" in item
        ),
        "",
    )
    return final if isinstance(final, str) else ""


@lru_cache(maxsize=1)
def _judge_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=os.environ["LLM_PROXY_URL"],
        api_key=os.environ["LLM_PROXY_MASTER_KEY"],
    )


async def _judge(question: str, gold: str, generated: str) -> bool:
    response = await _judge_client().chat.completions.create(
        model=os.getenv("JUDGE_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8"),
        messages=[
            {"role": "system", "content": _JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"QUESTION: {question}\n"
                    f"GOLD ANSWER: {gold}\n"
                    f"GENERATED ANSWER: {generated}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=100,
        extra_body={"enable_thinking": False},
    )
    message = response.choices[0].message.model_dump()
    reasoning = message.get("reasoning") or ""
    content = message.get("content") or ""
    raw = (reasoning + "\n" + content).strip() if reasoning else content
    verdict = _VERDICT_RE.search(raw)
    return bool(verdict and verdict.group(1).lower() == "true")


async def verify_answer(question: str, output: str, gold: str) -> bool:
    generated = _extract_final_answer(output)
    if generated and _inclusion_match(gold, generated):
        return True
    if not generated:
        return False
    return await _judge(question, gold, generated)
