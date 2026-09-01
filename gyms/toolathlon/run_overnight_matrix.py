"""Run the four September 2026 Toolathlon evaluation modes unattended."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "gyms" / "toolathlon"
MATRIX_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-overnight-four-modes")
MATRIX_DIR = ARTIFACTS / "matrices" / MATRIX_ID
E4B_DIR = Path("/home/matrosov/models/gemma-4-E4B-it")
GEMMA_26B_DIR = Path("/home/matrosov/models/gemma-4-26B-A4B-it")
PYTHON = sys.executable


def write_state(state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = MATRIX_DIR / ".state.json.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, MATRIX_DIR / "state.json")


def run_logged(name: str, command: list[str], env: dict[str, str]) -> int:
    path = MATRIX_DIR / f"{name}.log"
    with path.open("ab") as log:
        log.write(("COMMAND " + " ".join(command) + "\n").encode())
        log.flush()
        return subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode


def wait_for_model(port: int, model: str, process: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"model server exited; inspect {log}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=2
            ) as response:
                ids = {item["id"] for item in json.load(response)["data"]}
            if model in ids:
                return
        except (OSError, KeyError, json.JSONDecodeError):
            pass
        time.sleep(1)
    raise TimeoutError(f"model {model} was not ready within 1800 seconds")


def stop_group(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def batch_command(*extra: str, context_tokens: int = 256000) -> list[str]:
    return [
        PYTHON,
        str(ROOT / "gyms/toolathlon/run.py"),
        "--all-valid",
        "-n",
        "3",
        "--purpose",
        "evaluation",
        "--subagent-provider",
        "vllm",
        "--subagent-gpu",
        "6",
        "--vllm-data-parallel-size",
        "1",
        "--vllm-gpu-memory-utilization",
        "0.6",
        "--vllm-max-model-len",
        str(context_tokens),
        "--subagent-recursion-limit",
        "410",
        "--concurrency",
        "4",
        "--container-slots",
        "4",
        "--agent-timeout",
        "1800",
        "--episode-timeout",
        "2400",
        *extra,
    ]


def main() -> None:
    MATRIX_DIR.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["PATH"] = f"{Path(PYTHON).resolve().parent}:/home/matrosov/.local/bin:" + env.get(
        "PATH", ""
    )
    env["XDG_RUNTIME_DIR"] = "/run/user/1006"
    env["DOCKER_HOST"] = "unix:///run/user/1006/podman/podman.sock"
    state = {
        "matrix_id": MATRIX_ID,
        "status": "running",
        "gpu_policy": {"subagent": 6, "local_decomposer": 7},
        "gpu_memory_utilization": 0.6,
        "context_policy": {
            "qwen_subagent": 256000,
            "gemma_e4b_subagent_native_limit": 131072,
            "gemma_26b_decomposer": 256000,
        },
        "modes": {},
    }
    write_state(state)

    modes = [
        (
            "01-qwen35-4b-simple-nonthinking",
            batch_command(
                "--agent-mode", "simple",
                "--subagent-model", "Qwen/Qwen3.5-4B",
            ),
            env,
        ),
    ]

    for name, command, mode_env in modes:
        state["modes"][name] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
        write_state(state)
        code = run_logged(name, command, mode_env)
        state["modes"][name].update(status="completed" if code == 0 else "failed", returncode=code)
        write_state(state)

    download_code = run_logged(
        "02-gemma-e4b-download",
        [
            str(Path(PYTHON).with_name("hf")), "download",
            "google/gemma-4-E4B-it", "--local-dir", str(E4B_DIR),
        ],
        env,
    )
    if download_code != 0:
        state["modes"]["02-gemma4-e4b-simple-thinking"] = {
            "status": "failed", "returncode": download_code, "error": "model download failed"
        }
    else:
        name = "02-gemma4-e4b-simple-thinking"
        state["modes"][name] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
        write_state(state)
        code = run_logged(
            name,
            batch_command(
                "--agent-mode", "simple",
                "--subagent-model", str(E4B_DIR),
                context_tokens=131072,
            ),
            env,
        )
        state["modes"][name].update(status="completed" if code == 0 else "failed", returncode=code)
    write_state(state)

    proxy_log = (MATRIX_DIR / "lmrouter-connect-proxy.log").open("ab")
    proxy = subprocess.Popen(
        [PYTHON, str(ROOT / "gyms/toolathlon/fixed_connect_proxy.py")],
        cwd=ROOT,
        stdout=proxy_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        time.sleep(1)
        lm_env = {**env, "HTTPS_PROXY": "http://127.0.0.1:18042", "https_proxy": "http://127.0.0.1:18042"}
        name = "03-qwen36-35b-decomposer-qwen35-4b-subagents"
        state["modes"][name] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
        write_state(state)
        code = run_logged(
            name,
            batch_command(
                "--agent-mode", "decomposer",
                "--decomposer-provider", "lmrouter",
                "--decomposer-prompt", "teacher",
                "--model", "Qwen/Qwen3.6-35B-A3B-FP8",
                "--subagent-model", "Qwen/Qwen3.5-4B",
            ),
            lm_env,
        )
        state["modes"][name].update(status="completed" if code == 0 else "failed", returncode=code)
        write_state(state)
    finally:
        stop_group(proxy)
        proxy_log.close()

    server_log_path = MATRIX_DIR / "04-gemma26b-decomposer-vllm.log"
    server_log = server_log_path.open("ab")
    server_env = {**env, "CUDA_VISIBLE_DEVICES": "7"}
    server = subprocess.Popen(
        [
            str(Path(PYTHON).with_name("vllm")), "serve", str(GEMMA_26B_DIR),
            "--served-model-name", "google/gemma-4-26B-A4B-it",
            "--host", "0.0.0.0", "--port", "8040",
            "--max-model-len", "256000", "--gpu-memory-utilization", "0.6",
            "--language-model-only", "--enable-auto-tool-choice",
            "--tool-call-parser", "gemma4", "--reasoning-parser", "gemma4",
        ],
        cwd=ROOT,
        env=server_env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    name = "04-gemma4-26b-decomposer-gemma4-e4b-subagents"
    try:
        wait_for_model(8040, "google/gemma-4-26B-A4B-it", server, server_log_path)
        state["modes"][name] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
        write_state(state)
        code = run_logged(
            name,
            batch_command(
                "--agent-mode", "decomposer",
                "--decomposer-provider", "vllm",
                "--decomposer-base-url", "http://127.0.0.1:8040/v1",
                "--decomposer-prompt", "teacher",
                "--model", "google/gemma-4-26B-A4B-it",
                "--subagent-model", str(E4B_DIR),
                context_tokens=131072,
            ),
            env,
        )
        state["modes"][name].update(status="completed" if code == 0 else "failed", returncode=code)
    except BaseException as error:
        state["modes"][name] = {"status": "failed", "error": repr(error)}
    finally:
        stop_group(server)
        server_log.close()
        write_state(state)

    state["status"] = "completed"
    write_state(state)


if __name__ == "__main__":
    main()
