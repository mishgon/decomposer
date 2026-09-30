"""veRL supplies policy tokens; the shared Gym owns tools and native scoring."""

import asyncio
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import message_to_dict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

from decomposer.core import create_decomposer_agent
from gyms.toolathlon_gym.episode import Episode
from gyms.toolathlon_gym.run import invoke_and_capture
from rl.toolathlon_gym.policy import PolicyTokens, RolloutBudgetExceeded, VerlChatModel


async def run_blocking(function):
    """Let container operations finish before cleanup on cancellation."""
    operation = asyncio.create_task(asyncio.to_thread(function))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        await operation
        raise


@register("toolathlon_decomposer")
class ToolathlonAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        task = kwargs["extra_info"]["task_id"]
        episode_id = uuid4().hex
        directory = Path(os.environ["RL_ARTIFACTS"]) / "episodes" / episode_id
        episode = Episode(task, directory,
                          image=os.environ.get("RL_GYM_IMAGE", "decomposer-toolathlon:latest"))
        started = time.time()
        state, tokens, stop_reason = {}, None, "startup_error"
        try:
            await run_blocking(episode.start)

            async def generate(ids, params):
                return await self.server_manager.generate(request_id=episode_id, prompt_ids=ids,
                                                          sampling_params=params)

            tokens = PolicyTokens(self.tokenizer, generate,
                                  {**sampling_params, "presence_penalty": 1.5,
                                   "min_p": 0.0, "repetition_penalty": 1.0},
                                  prompt_budget=self.rollout_config.prompt_length,
                                  response_budget=self.rollout_config.response_length,
                                  log_path=directory / "policy_calls.jsonl")
            agent = create_decomposer_agent(
                decomposer_model=VerlChatModel(tokens=tokens),
                subagent_types=[{
                    "subagent_type_id": "qwen_3_5_4b_unlooped_non_thinking",
                    "description": "Qwen3.5-4B unlooped non-thinking agent with all task tools.",
                    "assistant_id": "qwen_3_5_4b_unlooped_non_thinking", "url": episode.url,
                }],
                subagent_recursion_limit=410, checkpointer=InMemorySaver())
            config = {"recursion_limit": 410, "configurable": {"thread_id": episode_id}}
            state, error = await invoke_and_capture(
                agent, {"messages": [{"role": "user", "content": episode.runtime["task_config"]["task_str"]}]},
                config, float(os.environ.get("RL_EPISODE_TIMEOUT", "2700")), episode.url)
            stop_reason = type(error).__name__ if error else "finished"
            # Never score a workspace that remote workers could still be changing.
            if "subagent_shutdown_error" in state:
                raise RuntimeError(state["subagent_shutdown_error"]) from error
            if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise error
            evaluation = await run_blocking(episode.score)
            if error and not isinstance(error, (RolloutBudgetExceeded, GraphRecursionError, TimeoutError)):
                raise error
            if not tokens.response_ids or not any(tokens.mask):
                raise RuntimeError("Episode produced no policy tokens")
            return AgentLoopOutput(
                prompt_ids=tokens.prompt_ids, response_ids=tokens.response_ids,
                response_mask=tokens.mask, response_logprobs=tokens.logprobs,
                reward_score=evaluation["reward"], num_turns=tokens.turns, metrics={},
                extra_fields={**tokens.extra, "task_id": task, "stop_reason": stop_reason,
                              "elapsed_seconds": time.time() - started})
        except BaseException as error:
            stop_reason = type(error).__name__
            raise
        finally:
            try:
                if directory.exists():
                    trace = {"task_id": task, "episode_id": episode_id,
                             "split": kwargs["extra_info"].get("split"),
                             "data_source": kwargs.get("data_source"),
                             "elapsed_seconds": time.time() - started, "stop_reason": stop_reason,
                             "messages": [message_to_dict(m) for m in state.get("messages", [])],
                             "subagent_runs": state.get("subagent_runs", {}),
                             "subagents": state.get("subagents", {}),
                             "subagent_shutdown": state.get("subagent_shutdown", []),
                             "subagent_shutdown_error": state.get("subagent_shutdown_error")}
                    if tokens is not None:
                        trace.update(policy_calls=tokens.calls, prompt_ids=tokens.prompt_ids,
                                     response_ids=tokens.response_ids, response_mask=tokens.mask,
                                     response_logprobs=tokens.logprobs, weight_versions=tokens.extra)
                    (directory / "trace.json").write_text(json.dumps(trace, default=str))
            finally:
                if directory.exists():
                    await run_blocking(episode.close)
