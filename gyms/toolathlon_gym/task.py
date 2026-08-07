import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def prepare_task(
    task_name: str,
    output_dir: Path,
    *,
    toolathlon_root: Path,
    run_preprocess: bool = True,
) -> Path:
    tasks_dir = (toolathlon_root / "tasks" / "finalpool").resolve()
    task_dir = (tasks_dir / task_name).resolve()
    if task_dir.parent != tasks_dir or not task_dir.is_dir():
        raise ValueError(f"Unknown Toolathlon task: {task_name!r}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"Toolathlon output directory must be empty: {output_dir}")

    os.chdir(toolathlon_root)
    sys.path.insert(0, str(toolathlon_root))

    from utils.data_structures.task_config import TaskConfig

    task_config = TaskConfig.build(
        task_name,
        agent_short_name="decomposer",
        global_task_config={
            "direct_to_dumps": True,
            "dump_path": str(output_dir),
        },
        single_turn_mode=True,
    )
    task_config.ensure_directories()
    workspace = task_config.agent_workspace_path
    if task_config.initialization.workspace is not None:
        for item in Path(task_config.initialization.workspace).iterdir():
            target = workspace / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    task_directories = {
        "arxiv_local": "arxiv_local_storage",
        "memory": "memory",
        "playwright_with_chunk": ".playwright_output",
    }
    for server, directory in task_directories.items():
        if server in task_config.needed_mcp_servers:
            (workspace / directory).mkdir(exist_ok=True)

    if run_preprocess and task_config.initialization.process_command is not None:
        subprocess.run(
            [
                *shlex.split(task_config.initialization.process_command),
                "--agent_workspace",
                str(workspace),
                "--launch_time",
                " ".join(task_config.launch_time.split()[:2]),
            ],
            check=True,
        )

    runtime_path = output_dir / "runtime.json"
    runtime_path.write_text(
        json.dumps({"task_config": task_config.to_dict()}, indent=2),
        encoding="utf-8",
    )
    return runtime_path


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()

    if args.command == "serve":
        task_name = os.environ.get("TOOLATHLON_TASK")
        if not task_name:
            raise RuntimeError("Set TOOLATHLON_TASK to a task name.")

        os.environ.setdefault("PYTHON_BIN", "/opt/venv/bin/python3")
        toolathlon_root = Path(os.environ.get("TOOLATHLON_ROOT", "/workspace"))
        data_dir = Path(os.environ.get("TOOLATHLON_DATA_DIR", "/artifacts/data"))
        runtime_path = prepare_task(
            task_name,
            data_dir,
            toolathlon_root=toolathlon_root,
            run_preprocess=not args.skip_preprocess,
        )
        print(f"Prepared {task_name}: {runtime_path}", flush=True)

        server = Path(__file__).parent / "subagents" / "serve.sh"
        os.execv(server, [str(server)])


if __name__ == "__main__":
    main()
