"""Score student samples with the teacher's per-token log-probs.

The teacher must share the student's tokenizer: its log-prob of each generated
student token is read from vLLM's `prompt_logprobs` on the student's exact token ids
(`/v1/completions`, `prompt=<ids>`, `max_tokens=1`, `prompt_logprobs=0`). The shared
LLM proxy passes both through; `return_token_ids` makes it echo the ids it scored,
which is checked against the request. Any vLLM-compatible base URL works.

    python -m training.opd.teacher probe --model Qwen/Qwen3.8-Flash-Next-NVFP4
    python -m training.opd.teacher score ROUND/samples.jsonl --output ROUND/scored.jsonl --model ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from training.opd.samples import Sample, read_samples

DEFAULT_BASE_URL_ENV = "LLM_PROXY_URL"
DEFAULT_API_KEY_ENV = "LLM_PROXY_MASTER_KEY"
STUDENT_TOKENIZER = "Qwen/Qwen3.5-4B"
STUDENT_TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class TeacherError(RuntimeError):
    pass


@dataclass(frozen=True)
class TeacherEndpoint:
    base_url: str
    api_key: str
    model: str
    verify_tls: bool = False
    timeout_seconds: float = 900.0
    max_retries: int = 4

    @classmethod
    def from_env(
        cls,
        *,
        model: str,
        base_url_env: str = DEFAULT_BASE_URL_ENV,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        verify_tls: bool = False,
    ) -> "TeacherEndpoint":
        missing = [name for name in (base_url_env, api_key_env) if not os.environ.get(name)]
        if missing:
            raise TeacherError(f"{' and '.join(missing)} must be set (source ~/.secrets/decomposer.env)")
        return cls(
            base_url=os.environ[base_url_env].rstrip("/"),
            api_key=os.environ[api_key_env],
            model=model,
            verify_tls=verify_tls,
        )


def token_logprobs(response: dict[str, Any], input_ids: Sequence[int]) -> list[float | None]:
    """Per-position log-prob of each actual input token (None at position 0)."""
    choice = response["choices"][0]
    echoed = choice.get("prompt_token_ids")
    if echoed is not None and list(echoed) != list(input_ids):
        raise TeacherError("the teacher scored different token ids than it was sent")
    entries = choice.get("prompt_logprobs")
    if entries is None or len(entries) != len(input_ids):
        raise TeacherError(
            f"expected {len(input_ids)} prompt_logprobs entries, got "
            f"{None if entries is None else len(entries)}"
        )
    values: list[float | None] = []
    for position, (entry, token) in enumerate(zip(entries, input_ids, strict=True)):
        if entry is None:
            if position != 0:
                raise TeacherError(f"missing prompt log-prob at position {position}")
            values.append(None)
            continue
        scored = entry.get(str(token))
        if scored is None:
            raise TeacherError(f"prompt log-prob at position {position} is not keyed by the actual token")
        values.append(float(scored["logprob"]))
    return values


class TeacherScorer:
    def __init__(self, endpoint: TeacherEndpoint, *, concurrency: int = 8) -> None:
        self.endpoint = endpoint
        self.semaphore = asyncio.Semaphore(concurrency)

    async def _post(self, client: httpx.AsyncClient, path: str, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.endpoint.max_retries + 1):
            try:
                response = await client.post(f"{self.endpoint.base_url}/{path}", json=body)
            except httpx.TransportError:
                if attempt == self.endpoint.max_retries:
                    raise
            else:
                if response.status_code < 400:
                    return response.json()
                if response.status_code not in RETRYABLE_STATUSES or attempt == self.endpoint.max_retries:
                    raise TeacherError(f"teacher returned {response.status_code}: {response.text[:300]}")
            await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.endpoint.timeout_seconds),
            verify=self.endpoint.verify_tls,
            headers={"Authorization": f"Bearer {self.endpoint.api_key}"},
            trust_env=True,
        )

    async def score_ids(self, client: httpx.AsyncClient, input_ids: Sequence[int]) -> list[float | None]:
        body = {
            "model": self.endpoint.model,
            "prompt": list(input_ids),
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 0,
            "return_token_ids": True,
        }
        async with self.semaphore:
            response = await self._post(client, "completions", body)
        return token_logprobs(response, input_ids)

    async def score_sample(self, client: httpx.AsyncClient, sample: Sample) -> list[float]:
        values = await self.score_ids(client, sample.input_ids)
        return [values[position] for position in sample.generated_positions()]  # type: ignore[misc]


async def _score_file(
    scorer: TeacherScorer, samples: list[Sample], output: Path
) -> dict[str, Any]:
    done: set[str] = set()
    if output.is_file():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["sample_id"])
    pending = [sample for sample in samples if sample.sample_id not in done]
    failures: list[dict[str, str]] = []
    lock = asyncio.Lock()
    output.parent.mkdir(parents=True, exist_ok=True)
    async with scorer.client() as client:
        with output.open("a", encoding="utf-8") as handle:

            async def run(sample: Sample) -> None:
                try:
                    teacher = await scorer.score_sample(client, sample)
                except (TeacherError, httpx.HTTPError) as error:
                    failures.append({"sample_id": sample.sample_id, "error": str(error)[:300]})
                    return
                record = {**asdict(sample), "teacher_logprobs": teacher}
                async with lock:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()

            await asyncio.gather(*(run(sample) for sample in pending))
    return {"samples": len(samples), "already_scored": len(done), "newly_scored": len(pending) - len(failures),
            "failures": failures[:20], "failed": len(failures)}


def score_file(endpoint: TeacherEndpoint, samples_path: Path, output: Path, *, concurrency: int) -> dict[str, Any]:
    """Score every sample not yet in `output`; appends, so an interrupted run resumes."""
    return asyncio.run(_score_file(TeacherScorer(endpoint, concurrency=concurrency), read_samples(samples_path), output))


async def _probe(endpoint: TeacherEndpoint, tokenizer: Any) -> dict[str, Any]:
    scorer = TeacherScorer(endpoint, concurrency=1)
    messages = [
        {"role": "system", "content": "You are a manager agent."},
        {"role": "user", "content": "Count the orders over $1,200 — 直接回答."},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
            "name": "spawn_subagent",
            "arguments": {"subagent_type_id": "subagent_non_thinking", "prompt": "List the orders."}}}]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, chat_template_kwargs={"enable_thinking": False})
    student_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    report: dict[str, Any] = {"model": endpoint.model, "probe_tokens": len(student_ids)}
    async with scorer.client() as client:
        models = await client.get(f"{endpoint.base_url}/models")
        listed = {entry.get("id"): entry for entry in models.json().get("data", [])}
        report["model_listed"] = endpoint.model in listed
        report["max_model_len"] = (listed.get(endpoint.model) or {}).get("max_model_len")
        response = await scorer._post(client, "completions", {
            "model": endpoint.model, "prompt": text, "max_tokens": 1, "temperature": 0.0,
            "prompt_logprobs": 0, "return_token_ids": True,
        })
        teacher_ids = response["choices"][0].get("prompt_token_ids")
        report["tokenizer_identical"] = teacher_ids == student_ids
        values = await scorer.score_ids(client, student_ids)
        report["token_id_scoring"] = len(values) == len(student_ids) and all(v is not None for v in values[1:])
    report["ok"] = bool(report["model_listed"] and report["tokenizer_identical"] and report["token_id_scoring"])
    return report


def probe(endpoint: TeacherEndpoint, *, tokenizer_name: str, tokenizer_revision: str | None) -> dict[str, Any]:
    """Fail fast unless the endpoint can serve as an OPD teacher for this student."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=tokenizer_revision)
    return asyncio.run(_probe(endpoint, tokenizer))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("probe", "score"))
    parser.add_argument("samples", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url-env", default=DEFAULT_BASE_URL_ENV)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--verify-tls", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--student-tokenizer", default=STUDENT_TOKENIZER)
    parser.add_argument("--student-tokenizer-revision", default=STUDENT_TOKENIZER_REVISION)
    args = parser.parse_args(argv)
    endpoint = TeacherEndpoint.from_env(
        model=args.model, base_url_env=args.base_url_env, api_key_env=args.api_key_env, verify_tls=args.verify_tls
    )
    if args.command == "probe":
        report = probe(endpoint, tokenizer_name=args.student_tokenizer, tokenizer_revision=args.student_tokenizer_revision)
        print(json.dumps(report))
        return 0 if report["ok"] else 1
    if args.samples is None or args.output is None:
        parser.error("score needs SAMPLES and --output")
    summary = score_file(endpoint, args.samples, args.output, concurrency=args.concurrency)
    print(json.dumps(summary))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
