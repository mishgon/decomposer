"""Student tokens come from veRL; task execution and judging stay in the Gym."""
import json
import os
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

from gyms.wideseek.run import episode
from gyms.wideseek.runtime import save
from opd.policy import PolicyTokens, VerlChatModel, is_episode_failure
from opd.synth.retry import check_episode, retry_episode


@cache
def tasks(path):
    return {t['task_id']: t for t in map(json.loads, Path(path).read_text().splitlines())}


@register('wideseek_decomposer')
class WideSeekAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        return await retry_episode(lambda: self._run_attempt(sampling_params, **kwargs))

    async def _run_attempt(self, sampling_params, **kwargs):
        task = tasks(str(Path(os.environ['OPD_DATA']) / 'tasks.jsonl'))[kwargs['extra_info']['task_id']]
        episode_id = uuid4().hex
        directory = Path(os.environ['OPD_ARTIFACTS']) / 'episodes' / episode_id
        directory.mkdir(parents=True)
        save(directory / "task.json", {"task_id": task["task_id"]})

        async def generate(ids, params):
            return await self.server_manager.generate(request_id=episode_id, prompt_ids=ids, sampling_params=params)

        tokens = PolicyTokens(self.tokenizer, generate,
            {**sampling_params, 'presence_penalty': 0.0, 'min_p': 0., 'repetition_penalty': 1.},
            prompt_budget=self.rollout_config.prompt_length,
            response_budget=self.rollout_config.response_length, log_path=directory / 'policy_calls.jsonl')
        args = SimpleNamespace(model_calls=None, output_tokens=None,
            worker_url=self.config.wideseek.worker_url, episode_timeout=self.config.wideseek.timeout,
            agent='decomposer')
        result = None
        try:
            result = await episode(task, 'decomposer', 1, directory, args, policy=VerlChatModel(tokens=tokens))
            save(directory / 'result.json', {**result, 'score': result['evaluation']['score']})
            check_episode(result, has_tokens=bool(tokens.response_ids and any(tokens.mask)))
            error_type = (result.get('error') or '').split(':', 1)[0]
            failure = {'error_type': error_type, 'error': result.get('error')}
            if result['status'] == 'error' and not is_episode_failure(failure):
                raise RuntimeError(f"Episode infrastructure failure: {result.get('error')}")
            if result['evaluation']['score'] is None:
                if result['status'] == 'timeout':
                    # A completed wall-clock budget has no final answer to grade.
                    reward = 0.0
                else:
                    raise RuntimeError('Native judge failed; see episode result')
            else:
                reward = result['evaluation']['score']
            if not tokens.response_ids or not any(tokens.mask):
                raise RuntimeError('Episode produced no student tokens')
            return AgentLoopOutput(prompt_ids=tokens.prompt_ids, response_ids=tokens.response_ids,
                response_mask=tokens.mask, response_logprobs=tokens.logprobs,
                reward_score=reward, num_turns=tokens.turns, metrics={},
                extra_fields={**tokens.extra, 'task_id': task['task_id'], 'stop_reason': result['status']})
        finally:
            save(directory / 'trajectory.json', {
                'task_id': task['task_id'], 'data_source': kwargs.get('data_source'),
                'prompt_ids': tokens.prompt_ids, 'response_ids': tokens.response_ids,
                'response_mask': tokens.mask, 'response_logprobs': tokens.logprobs,
                'weight_versions': tokens.extra, 'result': result})
