"""Launch OPD after a stock baseline and teacher token-alignment preflight."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from decomposer.agent_server import agent_server
from gyms.synth import ROOT
from opd.synth.prepare import prepare
from opd.teacher import score
from opd.launch import add_gpu_arguments, gpu_overrides
from opd.report import save_report


async def main(args, overrides):
    baseline = json.loads((args.baseline / "manifest.json").read_text())
    if baseline["status"] != "completed" or baseline["model"] != "vllm/qwen_3_5_4b_non_thinking":
        raise ValueError("Finish the stock Qwen3.5-4B baseline first")
    overrides = [*overrides, *gpu_overrides(args.gpus, args.training_gpus)]
    model = args.model.resolve()
    if args.resume_from:
        checkpoint = args.resume_from.resolve()
        from opd.checkpoint_offload import MARKER
        if (checkpoint / MARKER).exists():
            raise ValueError(f'Checkpoint is offloaded to S3. Restore first: python -m opd.checkpoint_offload restore {checkpoint}')
        names = ['data.pt'] + [f'actor/{kind}_world_size_{args.training_gpus}_rank_{rank}.pt'
            for rank in range(args.training_gpus) for kind in ('model', 'optim', 'extra_state')]
        for name in names:
            if not (checkpoint / name).is_file():
                raise ValueError(f"Incomplete resume checkpoint: {checkpoint / name}")
        overrides = [*overrides, 'trainer.resume_mode=resume_path',
                     f'trainer.resume_from_path={checkpoint}']
    if not (model / "config.json").exists():
        raise ValueError("Student checkpoint does not exist")
    if args.workplace_validation:
        from opd.workplace_assistant.preflight import check
        await check()
    for patch in (ROOT / "opd/patches").glob("*.patch"):
        subprocess.run(["git", "-C", str(ROOT / "external/verl"), "apply", "--recount",
                        "--reverse", "--check", str(patch)], check=True, capture_output=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = output / "data"
    prepare(data, demo_validation=args.demo_validation, workplace_validation=args.workplace_validation)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    os.environ.update({"OPD_ROOT": str(ROOT), "OPD_DATA": str(data), "OPD_ARTIFACTS": str(output),
        "MODEL_PATH": str(model), "SYNTH_ARTIFACT_ROOT": str(output),
        "SYNTH_WORKER_URL": f"http://127.0.0.1:{port}",
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, args.gpus)), "RAY_ADDRESS": "local",
        "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}:{ROOT / 'external/verl'}:{os.environ.get('PYTHONPATH', '')}",
        "TOKENIZERS_PARALLELISM": "false", "TENSORBOARD_DIR": str(output / "tensorboard")})
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    manifest = {"status": "preflight", "started_at": time.time(), "model": str(model),
                "baseline": str(args.baseline.resolve()), "gpus": args.gpus,
                "overrides": overrides, "pid": os.getpid()}
    if args.resume_from:
        manifest['resume_from'] = str(checkpoint)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
        ids = tokenizer.apply_chat_template([{"role": "user", "content": "Copy file A to B."},
                                            {"role": "assistant", "content": "I will delegate the copy."}],
            tokenize=True, return_dict=False, enable_thinking=False)
        await asyncio.wait_for(score(ids, tokenizer=tokenizer, output=output / "teacher-preflight.json"), 90)
        async with agent_server(ROOT / "gyms/synth/langgraph.json", port=port, n_jobs_per_worker=64,
                                python_executable=args.worker_python):
            command = [sys.executable, "-u", "-m", "opd.synth.trainer", "--config-path",
                       str(ROOT / "opd/synth"), "--config-name",
                       "single_gpu" if len(args.gpus) == 1 else "experiment",
                       "hydra.searchpath=[pkg://verl.trainer.config]", *overrides]
            manifest["status"] = "running"
            (output / "run.json").write_text(json.dumps(manifest, indent=2))
            with (output / "trainer.log").open("w") as log:
                child = await asyncio.create_subprocess_exec(*command, cwd=ROOT, stdout=log, stderr=log)
                try:
                    code = await child.wait()
                finally:
                    if child.returncode is None:
                        child.send_signal(signal.SIGINT)
                        await child.wait()
            if code:
                raise RuntimeError(f"Trainer exited {code}; see trainer.log")
            manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error=type(error).__name__)
        raise
    finally:
        manifest["finished_at"] = time.time()
        (output / "run.json").write_text(json.dumps(manifest, indent=2))
    save_report(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-validation", action="store_true", help="Add the held-out demo tasks in EN and RU")
    parser.add_argument("--workplace-validation", action="store_true", help="Add the frozen 20-task Workplace panel")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path.home() / "models/Qwen3.5-4B")
    parser.add_argument("--output", type=Path, required=True)
    add_gpu_arguments(parser)
    parser.add_argument("--resume-from", type=Path, help="Restore full veRL checkpoint into a new output directory")
    parser.add_argument("--worker-python", type=Path, default=ROOT / ".venv-workers/bin/python")
    args, overrides = parser.parse_known_args()
    asyncio.run(main(args, overrides))
