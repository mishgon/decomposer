"""Run the exact AgentLoop with a frozen vLLM token endpoint, before any update."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from training.toolathlon_gym.agent_loop import ToolathlonAgentLoop


class FrozenServer:
    def __init__(self, url, model="Qwen/Qwen3.5-4B", log_dir=None):
        self.model, self.log_dir = model, log_dir
        self.client = AsyncOpenAI(base_url=url, api_key="EMPTY", max_retries=0, timeout=300)

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        started = time.time()
        params = dict(sampling_params)
        top_k = params.pop("top_k", 20)
        min_p = params.pop("min_p", 0.0)
        repetition_penalty = params.pop("repetition_penalty", 1.0)
        response = await self.client.completions.create(
            model=self.model, prompt=prompt_ids, logprobs=1,
            extra_body={"return_token_ids": True, "top_k": top_k, "min_p": min_p,
                        "repetition_penalty": repetition_penalty}, **params)
        choice = response.choices[0]
        if self.log_dir is not None:
            with (self.log_dir / "episodes" / request_id / "policy_timing.jsonl").open("a") as stream:
                stream.write(json.dumps({"started_at": started, "finished_at": time.time(),
                    "prompt_tokens": len(prompt_ids), "output_tokens": len(choice.token_ids)}) + "\n")
        return SimpleNamespace(token_ids=choice.token_ids, log_probs=choice.logprobs.token_logprobs,
                               extra_fields={"min_global_steps": 0, "max_global_steps": 0})


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="canvas-assignment-stats")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["RL_ARTIFACTS"] = str(args.output.resolve())
    loop = object.__new__(ToolathlonAgentLoop)
    loop.tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("MODEL_PATH", str(Path.home() / "models/Qwen3.5-4B")))
    loop.server_manager = FrozenServer("http://127.0.0.1:8025/v1")
    loop.rollout_config = SimpleNamespace(prompt_length=4096, response_length=12288)
    result = await loop.run({"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                             "presence_penalty": 1.5}, extra_info={"task_id": args.task})
    (args.output / "rollout.json").write_text(json.dumps(result.model_dump()))
    print(json.dumps({"reward": result.reward_score, "policy_tokens": sum(result.response_mask),
                      "total_response_tokens": len(result.response_ids), **result.extra_fields}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
