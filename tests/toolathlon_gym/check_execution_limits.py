"""CPU-only integration: real tool startup, Python execution, and cgroup OOM containment."""
import argparse
import json
from pathlib import Path
from uuid import uuid4

from gyms.toolathlon_gym.episode import Episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episode = Episode("ppt-snowflake-executive", args.output)
    probe = "rl-memory-probe-" + uuid4().hex[:12]
    try:
        episode.start()
        for name, gb in ((episode.container, 16), (episode.pg, 2)):
            config = json.loads(episode.command("inspect", name).stdout)[0]["HostConfig"]
            assert config["Memory"] == gb * 1024**3, config["Memory"]
            assert config["MemorySwap"] == config["Memory"]
        workspace = episode.runtime["task_config"]["agent_workspace"]
        code = ("import asyncio; from gyms.toolathlon_gym.subagents.python_execute import make_python_execute; "
                f"print(asyncio.run(make_python_execute({workspace!r})('print(12345)')))" )
        result = episode.command("exec", episode.container, "/opt/venv/bin/python3", "-c", code)
        assert "12345" in result.stdout and "Return code: 0" in result.stdout, result.stdout
        oom_probe = ("import subprocess,sys,pathlib; "
                     "p=subprocess.run([sys.executable,'-c','x=bytearray(256*1024*1024)']); "
                     "print(pathlib.Path('/sys/fs/cgroup/memory.events').read_text()); "
                     "assert p.returncode == -9, p.returncode")
        result = episode.command("run", "--name", probe, "--network", "none",
                                 "--memory", "64m", "--memory-swap", "64m",
                                 "--entrypoint", "/opt/venv/bin/python3", episode.image,
                                 "-c", oom_probe, check=False)
        events = dict(line.split() for line in result.stdout.splitlines() if line.strip())
        assert result.returncode == 0 and int(events.get("oom_kill", 0)) >= 1, result
        print(json.dumps({"python_tool": "passed", "task_limit_gib": 16,
                          "database_limit_gib": 2, "oom_contained": True}))
    finally:
        episode.command("rm", "-f", probe, check=False)
        episode.close()


if __name__ == "__main__":
    main()
