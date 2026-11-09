"""Frozen Workplace panel: current checkpoint coordinator, fixed hosted workers."""
import json
import os
from pathlib import Path
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from decomposer.models import create_model
from gyms.workplace_assistant.in_process import episode
from opd.policy import PolicyTokens, VerlChatModel, is_episode_failure
from opd.workplace_assistant.preflight import WORKER_MODEL

from opd.synth.retry import check_episode, retry_episode


@register('workplace_decomposer')
class WorkplaceAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        return await retry_episode(lambda: self._run_attempt(sampling_params, **kwargs))

    async def _run_attempt(self, sampling_params, **kwargs):
        if kwargs['data_source'] != 'workplace-assistant/validation20':
            raise ValueError('Workplace panel is validation-only')
        task_id = int(kwargs['extra_info']['task_id'])
        panel = json.loads((Path(os.environ['OPD_ROOT']) /
                            'gyms/workplace_assistant/validation20.json').read_text())
        task = next(row for row in panel if row['id'] == task_id)
        episode_id = uuid4().hex
        directory = Path(os.environ['OPD_ARTIFACTS']) / 'workplace-episodes' / episode_id
        directory.mkdir(parents=True)
        async def generate(ids, params):
            return await self.server_manager.generate(request_id=episode_id, prompt_ids=ids,
                                                      sampling_params=params)
        tokens = PolicyTokens(self.tokenizer, generate, sampling_params,
            prompt_budget=self.rollout_config.prompt_length,
            response_budget=self.rollout_config.response_length,
            log_path=directory / 'policy_calls.jsonl')
        worker = create_model(WORKER_MODEL)
        try:
            result = await episode(task, VerlChatModel(tokens=tokens), worker)
        finally:
            worker.http_client.close()
            await worker.http_async_client.aclose()
        (directory / 'result.json').write_text(json.dumps(result, ensure_ascii=False))
        check_episode(result, has_tokens=bool(tokens.response_ids and any(tokens.mask)))
        if result['error_type'] is not None and not is_episode_failure(result):
            raise RuntimeError(f'Workplace validation failure: {result["error"]}; see {directory}')
        if not tokens.response_ids or not any(tokens.mask):
            raise RuntimeError('No Workplace validation student tokens produced')
        return AgentLoopOutput(prompt_ids=tokens.prompt_ids, response_ids=tokens.response_ids,
            response_mask=tokens.mask, response_logprobs=tokens.logprobs,
            reward_score=float(result['passed']), num_turns=tokens.turns,
            metrics={'passed': float(result['passed'])},
            extra_fields={**tokens.extra, 'task_id': str(task_id), 'stop_reason': result['status']})
