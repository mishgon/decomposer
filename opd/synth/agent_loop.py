"""Use the existing veRL token bridge; only the task and workers are synthetic."""
import json
import os
from pathlib import Path
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from gyms.synth.run import episode
from gyms.synth.task import make_task
from opd.policy import PolicyTokens, VerlChatModel
from opd.synth.retry import EpisodeTransportError, retry_episode


@register("synth_decomposer")
class SynthAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        return await retry_episode(lambda: self._run_attempt(sampling_params, **kwargs))

    async def _run_attempt(self, sampling_params, **kwargs):
        task_id = kwargs["extra_info"]["task_id"]
        split, index = task_id.split("-")
        task = make_task(split, int(index))
        episode_id = uuid4().hex
        directory = Path(os.environ["OPD_ARTIFACTS"]) / "episodes" / episode_id
        directory.parent.mkdir(parents=True, exist_ok=True)

        async def generate(ids, params):
            return await self.server_manager.generate(request_id=episode_id, prompt_ids=ids, sampling_params=params)

        tokens = PolicyTokens(self.tokenizer, generate,
            {**sampling_params, "presence_penalty": 1.5, "min_p": 0., "repetition_penalty": 1.},
            prompt_budget=self.rollout_config.prompt_length,
            response_budget=self.rollout_config.response_length, log_path=directory / "policy_calls.jsonl")
        result = None
        try:
            result = await episode(task, VerlChatModel(tokens=tokens), directory,
                                   self.config.synth.worker_url, timeout=180)
            if result["cleanup_errors"]:
                raise RuntimeError("Synthetic worker cleanup failed")
            if result.get("error_type") in {"ReadError", "WriteError", "ConnectError",
                                            "RemoteProtocolError", "ReadTimeout", "ConnectTimeout"}:
                raise EpisodeTransportError(f"{result['error_type']}; saved episode: {directory}")
            if result["status"] == "error" and not result["error"].startswith(
                    ("RolloutBudgetExceeded", "GraphRecursionError", "TimeoutError")):
                raise RuntimeError(f"Rollout infrastructure failure: {result['error']}")
            if not tokens.response_ids or not any(tokens.mask):
                raise RuntimeError("No student tokens produced")
            return AgentLoopOutput(prompt_ids=tokens.prompt_ids, response_ids=tokens.response_ids,
                response_mask=tokens.mask, response_logprobs=tokens.logprobs,
                reward_score=result["score"], num_turns=tokens.turns,
                metrics={"correct_and_parallel": float(result["passed"] and result["parallel"]),
                         "copies": float(result["copies"])},
                extra_fields={**tokens.extra, "task_id": task_id, "stop_reason": result["status"]})
        finally:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "trajectory.json").write_text(json.dumps({
                "task_id": task_id, "data_source": kwargs.get("data_source"),
                "prompt_ids": tokens.prompt_ids, "response_ids": tokens.response_ids,
                "response_mask": tokens.mask, "response_logprobs": tokens.logprobs,
                "weight_versions": tokens.extra, "result": result}))
