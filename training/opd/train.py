"""Train one OPD round on teacher-scored samples and export the next checkpoint.

    torchrun --nproc-per-node 3 -m training.opd.train \
        --config training/opd/configs/train_qwen35_4b_3gpu.yaml \
        --init <checkpoint> --samples ROUND/scored.jsonl --output-dir ROUND/train

One pass over the round's samples (each sample is used once, batch size one per
device). The forward computes logits only at the positions that predict generated
manager tokens (`logits_to_keep` with an index tensor), so a 64k-token context never
materialises [T, 248k] logits. FSDP settings follow training/sft; weights are kept
in fp32 with bf16 compute because OPD learning rates are small enough that bf16
master weights would round most updates away. The export is fp32 for the same
reason: the next round starts from it, and vLLM serves it with --dtype bfloat16.
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Qwen3.5's gated-delta layers run on flash-linear-attention. Its tilelang backward
# does not compile on Hertz-2 (the cu13 nvcc lacks <cuda/atomic>); the Triton kernels
# are what the SFT jobs use too (training/sft/run_train_jobs.py exports this).
os.environ.setdefault("FLA_TILELANG", "0")

import torch  # noqa: E402
import transformers  # noqa: E402
import yaml  # noqa: E402
from accelerate.utils import merge_fsdp_weights, save_fsdp_model  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)

from training.opd.loss import OPDLossConfig, opd_token_loss, selected_logprobs  # noqa: E402
from training.sft.train import (  # noqa: E402
    _configure_sdpa_backends,
    _is_rank_zero,
    _save_final_configuration,
    _world_size,
    _write_json,
)


class ScoredSamples(torch.utils.data.Dataset):
    def __init__(self, path: Path, *, max_length: int) -> None:
        self.records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            generated = sum(end - start for start, end in record["spans"])
            if len(record["teacher_logprobs"]) != generated or len(record["behavior_logprobs"]) != generated:
                raise ValueError(f"{record['sample_id']}: log-prob lengths do not match its spans")
            if len(record["input_ids"]) > max_length:
                raise ValueError(f"{record['sample_id']} is longer than max_length={max_length}")
            self.records.append(record)
        if not self.records:
            raise ValueError(f"no samples in {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def collate(batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError("OPD trains one sample per device step; set per_device_train_batch_size: 1")
    (record,) = batch
    input_ids = record["input_ids"]
    positions = [index for start, end in record["spans"] for index in range(start, end)]
    if not positions or positions[0] < 1:
        raise ValueError(f"{record['sample_id']}: a generated token needs at least one token of context")
    count = len(positions)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        # Logits at t-1 predict the token at t.
        "positions": torch.tensor([index - 1 for index in positions], dtype=torch.long),
        "targets": torch.tensor([input_ids[index] for index in positions], dtype=torch.long),
        "behavior_logprobs": torch.tensor(record["behavior_logprobs"], dtype=torch.float32),
        "teacher_logprobs": torch.tensor(record["teacher_logprobs"], dtype=torch.float32),
        "token_weight": torch.full((count,), float(record["token_weight"]), dtype=torch.float32),
        "reward_advantage": torch.full((count,), float(record["reward_advantage"]), dtype=torch.float32),
    }


class OPDTrainer(Trainer):
    _RATE_METRICS = ("reverse_kl", "advantage", "log_ratio_mean", "clip_fraction")

    def __init__(self, *args: Any, loss_config: OPDLossConfig, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
        self._sums: collections.defaultdict[str, float] = collections.defaultdict(float)
        self.round_metrics: dict[str, float] = {}
        self._round_sums: collections.defaultdict[str, float] = collections.defaultdict(float)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: ANN001
        outputs = model(input_ids=inputs["input_ids"], logits_to_keep=inputs["positions"], use_cache=False)
        logprobs = selected_logprobs(outputs.logits[0], inputs["targets"])
        loss, metrics = opd_token_loss(
            logprobs,
            inputs["behavior_logprobs"],
            inputs["teacher_logprobs"],
            inputs["token_weight"],
            inputs["reward_advantage"],
            self.loss_config,
        )
        tokens = metrics["tokens"]
        for sums in (self._sums, self._round_sums):
            sums["tokens"] += tokens
            for name in self._RATE_METRICS:
                sums[name] += metrics[name] * tokens
        return (loss, outputs) if return_outputs else loss

    def _reduce(self, sums: dict[str, float]) -> dict[str, float]:
        names = ["tokens", *self._RATE_METRICS]
        totals = torch.tensor([sums.get(name, 0.0) for name in names], device=self.args.device, dtype=torch.float64)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        tokens = float(totals[0])
        if not tokens:
            return {}
        return {"opd/tokens": tokens, **{f"opd/{name}": float(value) / tokens for name, value in zip(names[1:], totals[1:])}}

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self._sums:
            logs = {**logs, **self._reduce(self._sums)}
            self._sums.clear()
        super().log(logs, start_time)

    def finish_round_metrics(self) -> dict[str, float]:
        self.round_metrics = self._reduce(self._round_sums)
        return self.round_metrics


def load_policy(checkpoint: Path, *, dtype: torch.dtype, attn_implementation: str) -> tuple[Any, int]:
    """The checkpoint's own architecture, with any vision tower frozen.

    Qwen3.5 SFT exports are `Qwen3_5ForConditionalGeneration`. Loading them through
    AutoModelForCausalLM keeps only the text weights, so the export would drop the
    vision tower and change the layout vLLM expects. The manager never sees images:
    the tower is carried unchanged and gets no optimizer state.
    """
    architectures = getattr(AutoConfig.from_pretrained(checkpoint), "architectures", None) or []
    model_class = getattr(transformers, architectures[0], None) if architectures else None
    model = (model_class or AutoModelForCausalLM).from_pretrained(
        checkpoint, dtype=dtype, attn_implementation=attn_implementation
    )
    frozen = 0
    for name, parameter in model.named_parameters():
        if "visual" in name.split("."):
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    return model, frozen


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a mapping")
    return config


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--init", required=True, type=Path, help="checkpoint the round starts from")
    parser.add_argument("--samples", required=True, type=Path, help="teacher-scored samples")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    model_config = dict(config.get("model") or {})
    loss_config = OPDLossConfig(**(config.get("loss") or {}))
    training = dict(config.get("training") or {})
    run = dict(config.get("run") or {})

    expected_world_size = int(run.get("expected_world_size", 1))
    if _world_size() != expected_world_size:
        raise RuntimeError(
            f"This config expects {expected_world_size} processes, but WORLD_SIZE is {_world_size()}. "
            f"Launch it with torchrun --nproc-per-node={expected_world_size}."
        )
    global_batch_size = int(training.pop("global_batch_size"))
    if global_batch_size % expected_world_size:
        raise ValueError("training.global_batch_size must be divisible by the world size")
    max_length = int(run.get("max_length", 65536))
    _configure_sdpa_backends(run)

    output_dir = args.output_dir.resolve()
    final_dir = output_dir / "final"
    if final_dir.exists():
        raise FileExistsError(f"{final_dir} already exists; a round is trained once")
    dataset = ScoredSamples(args.samples, max_length=max_length)
    tokenizer = AutoTokenizer.from_pretrained(args.init)
    generation_config = GenerationConfig.from_pretrained(args.init)
    model, frozen_parameters = load_policy(
        args.init,
        dtype=getattr(torch, str(model_config.get("dtype", "float32"))),
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False
    model_class = type(model).__name__

    steps = math.ceil(len(dataset) / global_batch_size)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=global_batch_size // expected_world_size,
        num_train_epochs=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        seed=args.seed,
        data_seed=args.seed,
        **training,
    )
    trainer = OPDTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=collate,
        processing_class=tokenizer,
        loss_config=loss_config,
    )

    started = time.perf_counter()
    train_output = trainer.train()
    round_metrics = trainer.finish_round_metrics()
    elapsed = time.perf_counter() - started

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    sharded = fsdp_plugin is not None and "SHARDED_STATE_DICT" in str(fsdp_plugin.state_dict_type)
    if sharded:
        save_fsdp_model(fsdp_plugin, trainer.accelerator, trainer.model, str(final_dir))
    else:
        trainer.save_model(str(final_dir))
    if _is_rank_zero():
        _save_final_configuration(
            final_dir, model_config=trainer.model.config, tokenizer=tokenizer, generation_config=generation_config
        )
    if sharded:
        accelerator = trainer.accelerator
        accelerator.wait_for_everyone()
        accelerator.free_memory(trainer.model, trainer.optimizer)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        # Accelerate writes only on its main process, but this ends with a barrier
        # that every rank must enter.
        merge_fsdp_weights(
            final_dir / "pytorch_model_fsdp_0", final_dir, safe_serialization=True, remove_checkpoint_dir=True
        )
    if _is_rank_zero():
        _write_json(
            output_dir / "opd_summary.json",
            {
                "init": str(args.init),
                "samples": str(args.samples),
                "num_samples": len(dataset),
                "model_class": model_class,
                "frozen_parameters": frozen_parameters,
                "optimizer_steps": steps,
                "global_batch_size": global_batch_size,
                "loss": asdict(loss_config),
                "train_metrics": train_output.metrics,
                "round_metrics": round_metrics,
                "elapsed_seconds": elapsed,
                "final_model_dir": str(final_dir),
            },
        )


if __name__ == "__main__":
    main()
