"""Score student trajectories through an OpenAI-compatible prompt-logprob API."""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import time

import httpx


class HostTransport(httpx.AsyncHTTPTransport):
    """Optional DNS override without disabling TLS hostname verification."""

    def __init__(self, hostname, address):
        super().__init__()
        self.hostname, self.address = hostname, address

    async def handle_async_request(self, request):
        if request.url.host == self.hostname:
            request.headers['Host'] = self.hostname
            request.extensions['sni_hostname'] = self.hostname
            request.url = request.url.copy_with(host=self.address)
        return await super().handle_async_request(request)


def aligned_logprobs(response, token_ids, tokenizer=None):
    rows = response['choices'][0].get('prompt_logprobs')
    if not isinstance(rows, list) or len(rows) != len(token_ids):
        raise ValueError('Teacher omitted prompt scores or tokenization lengths differ')
    values = [0.0]  # First token has no preceding context; never a trained token.
    decoded = []
    for index, (token, row) in enumerate(zip(token_ids[1:], rows[1:]), 1):
        if not isinstance(row, dict) or str(token) not in row:
            raise ValueError(f'Teacher/student token mismatch at position {index}')
        if tokenizer is not None:
            part = row[str(token)].get('decoded_token')
            if not isinstance(part, str):
                raise ValueError(f'Teacher omitted decoded token at position {index}')
            decoded.append(part)
        value = row[str(token)]['logprob']
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f'Non-finite teacher logprob at position {index}')
        values.append(float(value))
    # vLLM decodes incrementally: UTF-8 byte tokens can emit '' followed by
    # the complete character. Compare the sequence, not individual fragments.
    if tokenizer is not None and ''.join(decoded) != tokenizer.decode(
            token_ids[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False):
        raise ValueError('Teacher/student token meaning mismatch')
    return values


async def score(token_ids, *, tokenizer, output, model=None):
    """Score exact generated IDs; decoding/re-encoding can change BPE boundaries."""
    model = model or os.environ['OPD_TEACHER_MODEL']
    payload = {'model': model, 'prompt': token_ids, 'prompt_logprobs': 0,
               'max_tokens': 1, 'temperature': 1.0}
    mapping = os.environ.get('OPD_TEACHER_HOST')
    transport = HostTransport(*mapping.split(':', 1)) if mapping else None
    base = os.environ['OPD_TEACHER_URL'].rstrip('/')
    key = os.environ.get('OPD_TEACHER_API_KEY') or os.environ['LLM_PROXY_MASTER_KEY']
    started = time.time()
    async with httpx.AsyncClient(transport=transport, timeout=300, trust_env=False) as client:
        response = await client.post(base + '/completions', json=payload,
                                     headers={'Authorization': f'Bearer {key}'})
        response.raise_for_status()
        raw = response.json()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'model': model, 'started_at': started,
                                 'elapsed_seconds': time.time() - started,
                                 'request': payload, 'response': raw}))
    return aligned_logprobs(raw, token_ids, tokenizer)


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    tokens = trace['prompt_ids'] + trace['response_ids']
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    values = asyncio.run(score(tokens, tokenizer=tokenizer, output=args.output))
    mask = trace['response_mask']
    response_values = values[len(trace['prompt_ids']):]
    assert len(mask) == len(response_values) and sum(mask) > 0
    print(json.dumps({'aligned_tokens': len(tokens), 'student_tokens': sum(mask),
                      'mean_teacher_logprob': sum(v for v, m in zip(response_values, mask) if m) / sum(mask)}))


if __name__ == '__main__':
    main()
