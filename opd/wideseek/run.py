"""Launch OPD after a starting-checkpoint baseline and teacher token-alignment preflight."""
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
from gyms.wideseek import REPO_ROOT as ROOT
from opd.wideseek.prepare import prepare
from opd.teacher import score
from opd.launch import add_gpu_arguments, gpu_overrides


async def main(args, overrides):
    trainer_env = None
    config_name = 'full'
    if bundle := os.environ.get('OPD_FAST_RUNTIME'):
        from opd.wideseek.fast_runtime import environment
        trainer_env = environment(Path(bundle))
        subprocess.run([sys.executable, '-m', 'opd.wideseek.fast_runtime'],
                       env=trainer_env, check=True)
        config_name = 'fast'
    baseline = json.loads((args.baseline / "manifest.json").read_text())
    if baseline["status"] != "completed" or baseline["model"] != "vllm/qwen_3_5_4b_non_thinking":
        raise ValueError("Finish the starting-checkpoint baseline first")
    training_gpus = args.training_gpus
    overrides = [*overrides, *gpu_overrides(args.gpus, training_gpus)]
    model = args.model.resolve()
    if baseline.get("student_model") != str(model):
        raise ValueError("Baseline must use the selected starting checkpoint")
    if args.resume_from:
        checkpoint = args.resume_from.resolve()
        from opd.checkpoint_offload import MARKER
        if (checkpoint / MARKER).exists():
            raise ValueError(f'Checkpoint is offloaded to S3. Restore first: python -m opd.checkpoint_offload restore {checkpoint}')
        names = ['data.pt'] + [f'actor/{kind}_world_size_{training_gpus}_rank_{rank}.pt'
            for rank in range(training_gpus) for kind in ('model', 'optim', 'extra_state')]
        for name in names:
            if not (checkpoint / name).is_file():
                raise ValueError(f"Incomplete resume checkpoint: {checkpoint / name}")
        overrides = [*overrides, 'trainer.resume_mode=resume_path',
                     f'trainer.resume_from_path={checkpoint}']
    if not (model / "config.json").exists():
        raise ValueError("Student checkpoint does not exist")
    from opd.workplace_assistant.preflight import check as check_workplace_worker
    await check_workplace_worker()
    for patch in (ROOT / "opd/patches").glob("*.patch"):
        subprocess.run(["git", "-C", str(ROOT / "external/verl"), "apply", "--recount",
                        "--reverse", "--check", str(patch)], check=True, capture_output=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = output / "data"
    panel = prepare(args.source, data)
    if baseline.get("source_sha256") != panel["source_sha256"]:
        raise ValueError("Baseline and training WideSeek source differ")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    os.environ.update({"OPD_ROOT": str(ROOT), "OPD_DATA": str(data), "OPD_ARTIFACTS": str(output),
        "MODEL_PATH": str(model), "WS_ARTIFACT_ROOT": str(output),
        "WS_WORKER_URL": f"http://127.0.0.1:{port}",
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, args.gpus)), "RAY_ADDRESS": "local",
        "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}:{ROOT / 'external/verl'}:{os.environ.get('PYTHONPATH', '')}",
        "TOKENIZERS_PARALLELISM": "false", "TENSORBOARD_DIR": str(output / "tensorboard")})
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    if trainer_env is not None:
        trainer_env = environment(Path(bundle))
    manifest = {"status": "preflight", "started_at": time.time(), "model": str(model),
                "baseline": str(args.baseline.resolve()), "gpus": args.gpus,
                "overrides": overrides, "pid": os.getpid()}
    manifest['config_name'] = config_name
    manifest['training_gpus'] = training_gpus
    manifest['rollout_gpus'] = max(1, len(args.gpus) - training_gpus)
    if trainer_env is not None:
        manifest['fast_runtime'] = json.loads((Path(bundle) / 'manifest.json').read_text())
    if args.resume_from:
        manifest['resume_from'] = str(checkpoint)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
        ids = tokenizer.apply_chat_template([{"role": "user", "content": "Copy file A to B."},
                                            {"role": "assistant", "content": "I will delegate the copy."}],
            tokenize=True, return_dict=False, enable_thinking=False)
        await asyncio.wait_for(score(ids, tokenizer=tokenizer, output=output / "teacher-preflight.json"), 90)
        async with agent_server(ROOT / "gyms/wideseek/langgraph.json", port=port, n_jobs_per_worker=64,
                                python_executable=args.worker_python):
            from opd.wideseek.preflight import check
            await check(model, output / "preflight", worker_url=os.environ["WS_WORKER_URL"],
                        search_url=os.environ["WS_SEARCH_URL"])
            command = [sys.executable, "-u", "-m", "opd.wideseek.trainer", "--config-path",
                       str(ROOT / "opd/wideseek"), "--config-name",
                       config_name,
                       "hydra.searchpath=[pkg://verl.trainer.config]", *overrides]
            manifest["status"] = "running"
            (output / "run.json").write_text(json.dumps(manifest, indent=2))
            with (output / "trainer.log").open("w") as log:
                child = await asyncio.create_subprocess_exec(*command, cwd=ROOT, stdout=log, stderr=log,
                                                             env=trainer_env)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned normalized WideSeek task JSONL")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path.home() / "models/Qwen3.5-4B")
    parser.add_argument("--output", type=Path, required=True)
    add_gpu_arguments(parser)
    parser.add_argument("--resume-from", type=Path, help="Restore full veRL checkpoint into a new output directory")
    parser.add_argument("--worker-python", type=Path, default=ROOT / ".venv-workers/bin/python")
    args, overrides = parser.parse_known_args()
    asyncio.run(main(args, overrides))
