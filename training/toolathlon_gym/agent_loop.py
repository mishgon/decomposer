"""veRL AgentLoop using the existing Decomposer and native Gym evaluator."""

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
from gyms.toolathlon_gym.cancel import cancel_subagents
from training.toolathlon_gym.policy import PolicyTokens, RolloutBudgetExceeded, VerlChatModel


@register("toolathlon_decomposer")
class ToolathlonAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        task = kwargs["extra_info"]["task_id"]
        episode_id = uuid4().hex
        directory = Path(os.environ["RL_ARTIFACTS"]) / "episodes" / episode_id
        episode = Episode(task, directory, subagent_port=int(os.environ.get("SUBAGENT_PORT", "8025")),
                          image=os.environ.get("RL_GYM_IMAGE", "decomposer-toolathlon-rl:latest"))
        started = time.time()
        startup = asyncio.create_task(asyncio.to_thread(episode.start))
        try:
            await asyncio.shield(startup)
        except asyncio.CancelledError:
            try:
                await startup
            finally:
                await asyncio.to_thread(episode.close)
            raise

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
            VerlChatModel(tokens=tokens),
            [{"subagent_type_id": "configured_non_thinking",
              "description": "Qwen3.5-4B non-thinking agent with all task tools.",
              "assistant_id": "configured_non_thinking", "url": episode.url}],
            subagent_recursion_limit=410, checkpointer=InMemorySaver())
        config = {"recursion_limit": 410, "configurable": {"thread_id": episode_id}}
        state, stop_reason, failure = {}, "finished", None
        try:
            try:
                state = await asyncio.wait_for(agent.ainvoke(
                    {"messages": [{"role": "user", "content": episode.runtime["task_config"]["task_str"]}]},
                    config=config), timeout=float(os.environ.get("RL_EPISODE_TIMEOUT", "2700")))
            except (RolloutBudgetExceeded, GraphRecursionError, TimeoutError) as error:
                stop_reason = type(error).__name__
                state = dict((await agent.aget_state(config)).values)
            except Exception as error:
                failure = error
                stop_reason = repr(error)
                state = dict((await agent.aget_state(config)).values)
            # Freeze unfinished subagents before native scoring of partial state.
            from langgraph_sdk import get_client
            client = get_client(url=episode.url)
            await cancel_subagents(client, state.get("subagent_runs", {}))
            evaluation = await asyncio.to_thread(episode.score)
            if failure is not None:
                raise failure
            if not tokens.response_ids or not any(tokens.mask):
                raise RuntimeError("Episode produced no policy tokens")
            return AgentLoopOutput(
                prompt_ids=tokens.prompt_ids, response_ids=tokens.response_ids,
                response_mask=tokens.mask, response_logprobs=tokens.logprobs,
                reward_score=evaluation["reward"], num_turns=tokens.turns, metrics={},
                extra_fields={**tokens.extra, "task_id": task, "stop_reason": stop_reason,
                              "elapsed_seconds": time.time() - started})
        finally:
            trace = {"task_id": task, "episode_id": episode_id,
                     "split": kwargs["extra_info"].get("split"),
                     "data_source": kwargs.get("data_source"),
                     "elapsed_seconds": time.time() - started, "stop_reason": stop_reason,
                     "messages": [message_to_dict(m) for m in state.get("messages", [])],
                     "subagent_runs": state.get("subagent_runs", {}),
                     "policy_calls": tokens.calls, "prompt_ids": tokens.prompt_ids,
                     "response_ids": tokens.response_ids, "response_mask": tokens.mask,
                     "response_logprobs": tokens.logprobs, "weight_versions": tokens.extra}
            (directory / "trace.json").write_text(json.dumps(trace, default=str))
            await asyncio.to_thread(episode.close)
