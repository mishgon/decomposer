"""Probe retrieval, worker graphs and teacher token alignment before allocating GPUs."""
import asyncio
from pathlib import Path

import httpx
from langgraph_sdk import get_client

from gyms.wideseek.runtime import model, close_model
from opd.teacher import score


async def check(model_path, output, *, worker_url, search_url):
    from transformers import AutoTokenizer
    async with httpx.AsyncClient(timeout=30, trust_env=False) as http:
        response = await http.post(search_url + '/retrieve', json={'queries': ['Paris'], 'topk': 1})
        response.raise_for_status()
        if not response.json()['result'][0]:
            raise ValueError('Retrieval returned no passages')
        response = await http.get(worker_url + '/ok')
        response.raise_for_status()
    client = get_client(url=worker_url)
    await client.assistants.get_graph('qwen_3_5_4b_unlooped_thinking')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ids = tokenizer.apply_chat_template([
        {'role': 'user', 'content': 'Research Paris. Réponds: café.'},
        {'role': 'assistant', 'content': 'I will search Wikipedia.'}],
        tokenize=True, return_dict=False, enable_thinking=False)
    values = await asyncio.wait_for(score(ids, tokenizer=tokenizer, output=output / 'teacher.json'), 90)
    subagent = model('qwen_3_5_4b_unlooped_thinking')
    try:
        reply = await asyncio.wait_for(subagent.ainvoke('Reply with OK.', max_tokens=512), 90)
        if not reply.content:
            raise ValueError('Subagent returned no content')
    finally:
        await close_model(subagent)
    return {'teacher_tokens': len(values), 'retrieval': 'ready', 'worker': 'ready', 'subagent': 'answered'}
