from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SERVED_MODEL = "google/gemma-4-26B-A4B-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve one Gemma vLLM replica per GPU and tunnel them to Hertz."
    )
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--local-port-start", type=int, default=8023)
    parser.add_argument("--remote-port-start", type=int, required=True)
    parser.add_argument("--hertz-host", required=True)
    parser.add_argument("--hertz-port", type=int, default=44444)
    parser.add_argument("--hertz-user", default="matrosov")
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=SERVED_MODEL)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.gpu_count < 1:
        parser.error("--gpu-count must be at least 1")
    return args


def wait_for_model(
    port: int,
    expected_model: str,
    timeout: float,
    process: subprocess.Popen[bytes] | None = None,
) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited with code {process.returncode} while starting port {port}"
            )
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                models = json.load(response)["data"]
            if expected_model in {model["id"] for model in models}:
                return
            last_error = RuntimeError(f"unexpected model list from {url}")
        except (OSError, KeyError, json.JSONDecodeError) as error:
            last_error = error
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become ready at {url}") from last_error


def vllm_command(args: argparse.Namespace, port: int) -> list[str]:
    return [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        args.model,
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
    ]


def tunnel_command(args: argparse.Namespace) -> list[str]:
    command = [
        "ssh",
        "-N",
        "-i",
        str(args.ssh_key),
        "-p",
        str(args.hertz_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=4",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={args.known_hosts}",
    ]
    for offset in range(args.gpu_count):
        command.extend(
            [
                "-R",
                (
                    f"127.0.0.1:{args.remote_port_start + offset}:"
                    f"127.0.0.1:{args.local_port_start + offset}"
                ),
            ]
        )
    command.append(f"{args.hertz_user}@{args.hertz_host}")
    return command


def terminate(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 30
    for process in reversed(processes):
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.ssh_key.is_file() or not args.known_hosts.is_file():
        raise FileNotFoundError("The dedicated tunnel key and known_hosts are required")

    processes: list[subprocess.Popen[bytes]] = []
    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for offset in range(args.gpu_count):
            port = args.local_port_start + offset
            log = (args.output_dir / f"vllm-gpu-{offset}.log").open("ab")
            environment = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(offset),
                # MLSpace's runtime image has the CUDA libraries but not nvcc.
                # FlashInfer sampling otherwise attempts a JIT build at warmup.
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
            }
            process = subprocess.Popen(
                vllm_command(args, port),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log.close()
            processes.append(process)

        for offset, process in enumerate(processes):
            if process.poll() is not None:
                raise RuntimeError(f"vLLM replica {offset} exited during startup")
            wait_for_model(
                args.local_port_start + offset,
                SERVED_MODEL,
                args.startup_timeout,
                process,
            )

        tunnel_log = (args.output_dir / "tunnel.log").open("ab")
        tunnel = subprocess.Popen(
            tunnel_command(args),
            stdout=tunnel_log,
            stderr=subprocess.STDOUT,
        )
        tunnel_log.close()
        processes.append(tunnel)
        time.sleep(3)
        if tunnel.poll() is not None:
            raise RuntimeError("SSH reverse tunnel exited during startup")

        ready = {
            "status": "ready",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gpu_count": args.gpu_count,
            "remote_ports": list(
                range(args.remote_port_start, args.remote_port_start + args.gpu_count)
            ),
            "model": SERVED_MODEL,
        }
        (args.output_dir / "ready.json").write_text(
            json.dumps(ready, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(ready), flush=True)

        while not stopping:
            failed = [process for process in processes if process.poll() is not None]
            if failed:
                raise RuntimeError(
                    f"{len(failed)} serving/tunnel process(es) exited unexpectedly"
                )
            time.sleep(2)
    finally:
        terminate(processes)


if __name__ == "__main__":
    main()
