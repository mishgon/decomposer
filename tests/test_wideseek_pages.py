import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from gyms.wideseek.pages import Pages, build


def test_disk_pages_preserve_unicode_duplicates_and_threaded_access(tmp_path):
    source = tmp_path / "pages.jsonl"
    rows = [{"url": "a", "text": "старое"}, {"url": "б", "text": "東京"},
            {"url": "a", "text": "новое"}]
    source.write_bytes(b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in rows))
    build(source)
    pages = Pages(source)
    assert len(pages) == 2
    assert pages.get("missing") is None
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(pages.get, ["б", "a"] * 10)) == [rows[1], rows[2]] * 10


def test_stale_or_failed_index_cannot_serve_changed_pages(tmp_path):
    source = tmp_path / "pages.jsonl"
    source.write_text('{"url":"a"}\n')
    build(source)
    original = source.with_suffix(".sqlite").read_bytes()
    source.write_text('invalid json\n')
    with pytest.raises(ValueError):
        Pages(source)
    with pytest.raises(ValueError):
        build(source)
    assert source.with_suffix(".sqlite").read_bytes() == original


def test_local_profiles_need_no_hosted_credentials():
    env = dict(os.environ, WS_MODEL_BACKEND="local")
    env.pop("LLM_PROXY_MASTER_KEY", None)
    subprocess.run([sys.executable, "-c", """
from gyms.wideseek import agents
from gyms.wideseek.runtime import model, close_model
import asyncio
assert agents.REACT_MODEL == 'vllm/qwen_3_5_4b_non_thinking'
assert agents.JUDGE_MODEL == 'vllm/qwen_3_5_4b_non_thinking'
for profile in (agents.REACT_MODEL, agents.DECOMPOSER_MODEL):
    policy = model(profile)
    assert policy.extra_body["chat_template_kwargs"]["enable_thinking"] is False
    assert policy.preserve_reasoning is False
    assert str(policy.openai_api_base) == 'http://127.0.0.1:8024/v1'
    asyncio.run(close_model(policy))
"""], env=env, check=True)
