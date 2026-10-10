"""Score student trajectories through an OpenAI-compatible prompt-logprob API."""

import argparse
import asyncio
import json
import math
from pathlib import Path
import time

from decomposer.models import create_model


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


async def score(token_ids, *, tokenizer, output):
    """Score exact generated IDs; decoding/re-encoding can change BPE boundaries."""
    model = create_model('lmrouter/qwen_3_8_flash_next_non_thinking')
    payload = {'model': model.model_name, 'prompt': token_ids, 'prompt_logprobs': 0,
               'max_tokens': 1, 'temperature': 1.0}
    started = time.time()
    record = {'model': model.model_name, 'started_at': started, 'request': payload, 'status': 'failed'}
    try:
        # Reuse the registry's authenticated client, tunnel and two SDK retries.
        response = await model.root_async_client.completions.create(
            model=model.model_name, prompt=token_ids, max_tokens=1, temperature=1.0,
            extra_body={'prompt_logprobs': 0})
        raw = response.model_dump()
        record['response'] = raw
        values = aligned_logprobs(raw, token_ids, tokenizer)
        record['status'] = 'completed'
        return values
    except Exception as error:
        record['error'] = {'type': type(error).__name__, 'http_status': getattr(error, 'status_code', None)}
        # OpenAI exceptions need HTTP response objects and cannot cross Ray workers.
        raise RuntimeError(f"Teacher scoring failed: {type(error).__name__}; "
                           f"HTTP {getattr(error, 'status_code', None)}; see {output}") from None
    finally:
        record['elapsed_seconds'] = time.time() - started
        try:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(record))
        finally:
            model.http_client.close()
            await model.http_async_client.aclose()


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
