"""Held-out demo validation through the same token bridge and in-process gym."""
import json
import os
from pathlib import Path
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase,AgentLoopOutput,register
from gyms.decomposer_demo.run import episode
from opd.policy import PolicyTokens,VerlChatModel


@register('demo_decomposer')
class DemoAgentLoop(AgentLoopBase):
    async def run(self,sampling_params,**kwargs):
        if kwargs['data_source'] not in {'decomposer-demo/en','decomposer-demo/ru'}:
            raise ValueError('Demo tasks are validation-only')
        info=kwargs['extra_info'];id=uuid4().hex
        directory=Path(os.environ['OPD_ARTIFACTS'])/'demo-episodes'/id
        directory.mkdir(parents=True)
        async def generate(ids,params):
            return await self.server_manager.generate(request_id=id,prompt_ids=ids,
                sampling_params={**params,'max_tokens':min(params['max_tokens'],4096)})
        tokens=PolicyTokens(self.tokenizer,generate,
            {**sampling_params,'presence_penalty':1.5,'min_p':0.,'repetition_penalty':1.},
            prompt_budget=self.rollout_config.prompt_length,response_budget=self.rollout_config.response_length,
            log_path=directory/'policy_calls.jsonl')
        result=await episode(info['task_id'],VerlChatModel(tokens=tokens),
                             language=info['language'])
        (directory/'result.json').write_text(json.dumps(result,ensure_ascii=False,default=str))
        if result['error_type'] not in {None,'RolloutBudgetExceeded','GraphRecursionError','TimeoutError'}:
            raise RuntimeError(f'Demo validation infrastructure failure: {result["error_type"]}; see {directory}')
        if not tokens.response_ids or not any(tokens.mask):
            raise RuntimeError('No validation student tokens produced')
        return AgentLoopOutput(prompt_ids=tokens.prompt_ids,response_ids=tokens.response_ids,
            response_mask=tokens.mask,response_logprobs=tokens.logprobs,reward_score=result['score'],
            num_turns=tokens.turns,metrics={'passed':float(result['passed'])},
            extra_fields={**tokens.extra,'task_id':info['task_id'],'stop_reason':result['status']})
