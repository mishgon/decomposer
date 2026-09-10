"""Run the exact AgentLoop with a frozen vLLM token endpoint, before any update."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from training.toolathlon_gym.agent_loop import ToolathlonAgentLoop


class FrozenServer:
    def __init__(self, url):
        self.client = AsyncOpenAI(base_url=url, api_key="EMPTY", max_retries=0, timeout=300)

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        params = dict(sampling_params)
        top_k = params.pop("top_k", 20)
        min_p = params.pop("min_p", 0.0)
        repetition_penalty = params.pop("repetition_penalty", 1.0)
        response = await self.client.completions.create(
            model="Qwen/Qwen3.5-4B", prompt=prompt_ids, logprobs=1,
            extra_body={"return_token_ids": True, "top_k": top_k, "min_p": min_p,
                        "repetition_penalty": repetition_penalty}, **params)
        choice = response.choices[0]
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
    loop.tokenizer = AutoTokenizer.from_pretrained("/home/matrosov/models/Qwen3.5-4B")
    loop.server_manager = FrozenServer("http://127.0.0.1:8025/v1")
    loop.rollout_config = SimpleNamespace(prompt_length=4096, response_length=12288)
    result = await loop.run({"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                             "presence_penalty": 1.5}, extra_info={"task_id": args.task})
    (args.output / "rollout.json").write_text(json.dumps(result.model_dump()))
    print(json.dumps({"reward": result.reward_score, "policy_tokens": sum(result.response_mask),
                      "total_response_tokens": len(result.response_ids), **result.extra_fields}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
