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
        assert "source_sha256" not in before
        with pytest.raises(FileExistsError):
            asyncio.run(prepare_run(args))
        monkeypatch.chdir(REPO_ROOT)
        args.output = tmp_path / "run"
        args.resume = True
        asyncio.run(prepare_collection(args))
        args.timeout += 1
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
