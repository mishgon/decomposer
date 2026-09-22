"""veRL teacher-client interface backed by lmrouter, with no teacher GPU pool."""
import os
from pathlib import Path
from types import SimpleNamespace

from .teacher import score


class HostedTeacherClient:
    def __init__(self, tokenizer_path):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None

    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        if sampling_params.get('prompt_logprobs') != 0:
            raise ValueError('Hosted teacher currently supports sampled-token OPD only')
        if any(kwargs.get(k) for k in ('image_data', 'video_data', 'audio_data')):
            raise ValueError('Toolathlon OPD supports text trajectories only')
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        output = Path(os.environ['RL_ARTIFACTS']) / 'teacher_calls' / f'{request_id}.json'
        values = await score(prompt_ids, tokenizer=self.tokenizer, output=output)
        # veRL indexes predictions by their preceding position, matching
        # vllm_rollout.utils.extract_prompt_logprobs (last position is padding).
        return SimpleNamespace(extra_fields={
            'prompt_ids': [[i] for i in prompt_ids[1:]] + [[0]],
            'prompt_logprobs': [[v] for v in values[1:]] + [[0.0]],
        })


class HostedTeacherManager:
    def __init__(self, config):
        self.client = HostedTeacherClient(config.actor_rollout_ref.model.path)

    def get_client(self):
        return {'default': self.client}
