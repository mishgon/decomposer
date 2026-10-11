import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gyms.wideseek import REPO_ROOT
from gyms.wideseek.run import create_parser, prepare_run
from gyms.wideseek.runtime import directory
from sft.wideseek.run import prepare_collection
import pytest


def test_collection_resume_and_paths_outside_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "tasks.jsonl"
    data.write_text(json.dumps({"task_id": "test", "question": "Q"}) + "\n")
    parser = create_parser()
    assert parser.parse_args(["--agent", "react", "--output", "run"]).data == (
        REPO_ROOT / "artifacts/gyms/wideseek/data/width/tasks.jsonl")
    args = parser.parse_args(["--agent", "react", "--output", "run", "--data", str(data)])
    args.mode = "simple"
    args.resume = False
    http = MagicMock()
    response = MagicMock()
    response.json.return_value = {"status": "ready"}
    http.__aenter__ = AsyncMock(return_value=SimpleNamespace(get=AsyncMock(return_value=response)))
    http.__aexit__ = AsyncMock()
    with patch("gyms.wideseek.run.model", return_value=SimpleNamespace(model_name="test")), \
            patch("gyms.wideseek.run.httpx.AsyncClient", return_value=http):
        asyncio.run(prepare_collection(args))
        before = json.loads((tmp_path / "run/manifest.json").read_text())
        assert before["revision"]
        assert before["settings"]["data_sha256"]
        assert before["settings"]["harness"]["source_sha256"]
        with pytest.raises(FileExistsError):
            asyncio.run(prepare_run(args))
        monkeypatch.chdir(REPO_ROOT)
        args.output = tmp_path / "run"
        args.resume = True
        asyncio.run(prepare_collection(args))
        args.episode_timeout += 1
        with pytest.raises(ValueError, match="settings or task data differ"):
            asyncio.run(prepare_collection(args))
    assert before == json.loads((tmp_path / "run/manifest.json").read_text())
    assert directory({"directory": str(tmp_path / "other-disk/episode")}) == tmp_path / "other-disk/episode"


def test_environment_requires_only_exported_credentials(tmp_path):
    env = {**os.environ, "HOME": str(tmp_path), "LLM_PROXY_MASTER_KEY": "test-key"}
    env.pop("LLM_PROXY_UNIX_SOCKET", None)
    script = REPO_ROOT / "gyms/wideseek/env.sh"
    subprocess.run(["bash", "-c", 'source "$1"', "bash", str(script)], env=env, check=True,
                   capture_output=True)
    env.pop("LLM_PROXY_MASTER_KEY")
    result = subprocess.run(["bash", "-c", 'source "$1"', "bash", str(script)], env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "Missing hosted model credential" in result.stderr
    assert "test-key" not in result.stderr


def test_setup_installs_only_into_gym_environment(tmp_path):
    gym = tmp_path / "gyms/wideseek"
    gym.mkdir(parents=True)
    script = gym / "setup.sh"
    script.write_text((REPO_ROOT / "gyms/wideseek/setup.sh").read_text())
    (tmp_path / "external/RLinf").mkdir(parents=True)
    root_venv = tmp_path / ".venv"
    root_venv.mkdir()
    (root_venv / "keep").write_text("unchanged")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
                       'with open(os.environ["COMMAND_LOG"], "a") as out:\n'
                       '    out.write(json.dumps(sys.argv[1:]) + "\\n")\n')
    fake_git = bin_dir / "git"
    fake_git.write_text('#!/bin/sh\nprintf "%s\\n" 64875d346d5cafb06c1112f563b20d2c6360bfae\n')
    fake_uv.chmod(0o755)
    fake_git.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    subprocess.run(["bash", str(script)], check=True, capture_output=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
             "UV_BIN": str(fake_uv), "COMMAND_LOG": str(log)})
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0] == ["venv", "--python", "3.12", "gyms/wideseek/.venv"]
    assert all(call[3] == "gyms/wideseek/.venv/bin/python" for call in calls[1:])
    assert list(root_venv.iterdir()) == [root_venv / "keep"]


@pytest.mark.parametrize('change', ['source', 'prompt', 'budget', 'revision', 'legacy'])
def test_resume_rejects_changed_harness_without_writing(tmp_path, monkeypatch, change):
    from copy import deepcopy
    from sft.wideseek import run as collector

    previous = {'revision': 'old', 'settings': {'harness': {
        'source_sha256': 'original', 'agent_system_prompt': 'original',
        'agent_run_budget_seconds': 60, 'agent_shutdown_grace_seconds': 60,
    }}}
    current = deepcopy(previous)
    if change == 'revision':
        current['revision'] = 'new'
    elif change == 'legacy':
        previous['settings'].pop('harness')
    else:
        field = {'source': 'source_sha256', 'prompt': 'agent_system_prompt',
                 'budget': 'agent_run_budget_seconds'}[change]
        current['settings']['harness'][field] = 'changed'
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(previous))
    before = path.read_bytes()
    monkeypatch.setattr(collector, 'describe_run', AsyncMock(return_value=([], tmp_path, current)))
    with pytest.raises(ValueError, match='harness revision'):
        asyncio.run(collector.prepare_collection(SimpleNamespace(resume=True)))
    assert path.read_bytes() == before


def test_harness_fingerprint_includes_uncommitted_prompt_changes(tmp_path, monkeypatch):
    from gyms.wideseek import run

    for relative in ('gyms/model_logging.py', 'gyms/usage.py', 'gyms/wideseek/langgraph.json',
                     'gyms/wideseek/prompts.py'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original')
    data = tmp_path / 'tasks.jsonl'
    data.write_text(json.dumps({'task_id': 'task', 'question': 'Q'}) + '\n')
    args = run.create_parser().parse_args(['--agent', 'react', '--output', str(tmp_path / 'run'),
                                         '--data', str(data)])
    args.mode = 'simple'
    monkeypatch.setattr(run, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *a, **kw: 'same-commit')
    monkeypatch.setattr(run, 'model', lambda _: SimpleNamespace(model_name='test'))
    http = MagicMock()
    response = MagicMock()
    response.json.return_value = {'status': 'ready'}
    http.__aenter__ = AsyncMock(return_value=SimpleNamespace(get=AsyncMock(return_value=response)))
    http.__aexit__ = AsyncMock()
    monkeypatch.setattr(run.httpx, 'AsyncClient', lambda **kwargs: http)
    original = asyncio.run(run.describe_run(args))[2]
    (tmp_path / 'gyms/wideseek/prompts.py').write_text('changed without a commit')
    changed = asyncio.run(run.describe_run(args))[2]
    assert original['revision'] == changed['revision']
    assert original['settings']['harness']['source_sha256'] != changed['settings']['harness']['source_sha256']
