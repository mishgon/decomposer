# Qwen3.5-4B SFT batch-size and FlashAttention-2 benchmark

> Historical benchmark: the final production decision is recorded in
> `docs/sft_qwen35_h100_grouped_batch_toolathlon_only_benchmark.md`.

Date: 2026-08-30

## Decision

Keep the production configuration at `per_device_train_batch_size: 1` and
`global_batch_size: 4` on four H100 80 GB GPUs. Batch size 2 per GPU fails on
the longest valid 32K examples with both SDPA and FlashAttention-2.

Do not add FlashAttention-2 as a required SFT dependency. On a representative
length- and environment-stratified workload it was 0.07% slower than SDPA and
used exactly the same peak PyTorch memory. The difference is measurement noise,
whereas the dependency introduces extra version and kernel-resolution failure
modes.

## Reproducibility

- Repository commit before the uncommitted benchmark helpers:
  `35dfc10bfb7f9fe87fb1da0efa80b85f60e1d176`.
- Configuration:
  `training/sft/configs/qwen35_4b_nonthinking_mixed_v3_gaia2_execution_110_n7_teacher_prompt_filtered_32k_full_4gpu.yaml`.
- Configuration SHA-256:
  `19502d70f7d7e00ebb284ec96ba429324bfba60d7cdc1435c98790fd9bbf3059`.
- Model: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, BF16, full-parameter FSDP,
  FSDP activation checkpointing, fused AdamW, and Liger fused linear cross
  entropy.
- Dataset: teacher-prompt mixed SFT v3 with Workplace, Toolathlon, and GAIA2;
  1,314 train and 127 validation records after runtime checks. Dataset
  fingerprint: `62d42708828c3dca8cde7dbc2bdfb6fd6a7eecf62050c4b126e00cfb72e62308`.
- Valid train lengths: minimum 2,734, median 3,872, p90 14,007, p95 18,837,
  maximum 32,626 tokens.
- Hardware: four of the eight otherwise-idle NVIDIA H100 80 GB HBM3 GPUs;
  driver 570.133.20. Every run explicitly used
  `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Runtime: Python 3.12.13, PyTorch 2.11.0+cu129, Transformers 5.14.1,
  TRL 1.9.2, and Liger Kernel 0.8.1.
- FlashAttention comparison runtime only: `kernels==0.15.2`,
  `kernels-data==0.16.1`, and `kernels-community/flash-attn2` version 1 as
  resolved by Transformers. These environment-only installs were not added to
  project requirements.
- Raw artifacts:
  `/home/sukhorukov/decomposer_artifacts/training/sft/benchmarks/qwen35-4b-h100-batch-fa2-20260830`.

The SDPA and FlashAttention representative runs have identical
`data_manifest.json` checksums:
`80d032d5d012fde8b87ee4d807425aeba2898becd2f50506841eb43a13a6c1a6`.
The benchmark selector deterministically preserves source proportions and
samples the complete length distribution. This avoids reporting a result that
only reflects the many short examples.

The common invocation was:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
.venv/bin/torchrun --standalone --nproc-per-node=4 \
  -m training.sft.train \
  --config training/sft/configs/qwen35_4b_nonthinking_mixed_v3_gaia2_execution_110_n7_teacher_prompt_filtered_32k_full_4gpu.yaml \
  --benchmark \
  --per-device-train-batch-size 1 \
  --global-batch-size 4 \
  --attn-implementation sdpa \
  --stratified-train-samples 192 \
  --max-eval-samples 1 \
  --output-dir /absolute/benchmark/output
```

For the alternative backend, `--attn-implementation` was changed to
`kernels-community/flash-attn2`. Capacity runs used
`--longest-train-samples 12` for batch 1 and 24 for batch 2, so each GPU saw
three of the longest records and both batch settings attempted the same three
optimizer steps.

## Results

### Full-32K capacity

| Backend | Per-GPU / global batch | Result | Tokens/s | Worst allocated / reserved |
| --- | ---: | --- | ---: | ---: |
| PyTorch SDPA | 1 / 4 | 3 steps passed | 1,702.61 | 54.38 / 62.76 GiB |
| PyTorch SDPA | 2 / 8 | OOM in first backward | n/a | 77.53 GiB allocated before failure |
| HF FlashAttention-2 kernel | 1 / 4 | 3 steps passed | 1,734.90 | 54.38 / 62.76 GiB |
| HF FlashAttention-2 kernel | 2 / 8 | OOM in first backward | n/a | 77.43 GiB allocated before failure |

The batch-2 failures were not allocator fragmentation: the processes had only
5-815 MiB free and approximately 76.7-77.5 GiB allocated. Stack traces point to
Qwen3.5's Torch linear-attention fallback (`torch_chunk_gated_delta_rule` and
the gated normalization/SILU path), not the eight full-attention layers.
FlashAttention therefore cannot raise the maximum batch size for this model.

### Representative workload

Both runs trained the same 192 examples for 48 optimizer steps and processed
1,200,953 input tokens. The sample spans 2,765-30,149 tokens with median 3,849,
p90 14,168, and p95 18,874.

| Backend | Runtime | Tokens/s | Train loss | Worst allocated / reserved |
| --- | ---: | ---: | ---: | ---: |
| PyTorch SDPA | 762.953 s | 1,574.08 | 0.669952 | 51.57 / 59.89 GiB |
| HF FlashAttention-2 kernel | 763.473 s | 1,573.01 | 0.669057 | 51.57 / 59.89 GiB |

FlashAttention-2 changed throughput by -0.068% and runtime by +0.068%. The
short longest-example capacity run showed a 1.90% gain, but the longer and more
representative paired run shows that this does not carry over to actual mixed
training. The small numerical loss difference is expected from changing the
attention kernel and does not indicate a quality difference in this throughput
benchmark.

## Problems encountered and resolutions

1. The current virtual environment has no `flash_attn`, no `nvcc` executable,
   and no compatible prebuilt PyPI wheel for Python 3.12 / Torch 2.11. The
   ordinary `flash_attention_2` backend therefore required an unsupported local
   source build. It was not used.
2. Transformers can load an official Hugging Face Hub kernel without compiling
   locally, but the literal `flash_attention_2` setting did not automatically
   fall back to it. Using the explicit
   `kernels-community/flash-attn2` attention implementation solved that part.
3. Installing the then-latest `kernels==0.16.1` failed because Transformers
   5.14.1 requires `kernels>=0.15.2,<0.16.0`. Pinning `kernels==0.15.2` made the
   backend load successfully. Transformers warned that Hub kernel version 1 was
   selected while version 3 exists; the supported default was retained rather
   than introducing another unpinned variable.
4. Qwen3.5-4B has 24 linear-attention layers and only eight full-attention
   layers. Both backends warned that `flash-linear-attention` and
   `causal-conv1d` are absent, so those 24 layers used the Transformers Torch
   fallback. This was held constant for a clean SDPA-versus-FlashAttention-2
   comparison. Benchmarking the model's linear-attention fast path is a more
   plausible follow-up optimization than adding FlashAttention-2.
5. Every `torchrun` rank repeats the full prepared-data validation with
   `num_proc=8`, and `SFTTrainer` then tokenizes its input again. A four-GPU run
   can therefore start 32 tokenization workers per preprocessing map. This adds
   noisy startup cost but is outside the timed `trainer.train()` interval and
   was kept unchanged between runs. It should be fixed separately by validating
   once and sharing cached prepared tokenization.
6. Running `pytest` directly from `.venv` failed at collection because `mls` is
   intentionally installed by the MLSpace requirements layer rather than the
   base training environment. Tests were run with
   `uv run --with-requirements training/sft/requirements-mlspace.txt pytest ...`.
7. `rg` and `jq` are not installed on this host. Diagnostic log inspection used
   `grep`, `sed`, and `awk`; no benchmark behavior was affected.
8. `torchrun` printed harmless IPv6 socket warnings on startup. Distributed
   training initialized and completed correctly, and all GPU processes released
   their memory after every successful and failed run.

## Uncommitted benchmark support

`training/sft/train.py` currently has uncommitted, benchmark-only CLI overrides
for batch size, attention implementation, longest-example selection, and
environment/length-stratified selection. Benchmark mode disables evaluation,
checkpointing, ClearML, and final export, resets CUDA peak counters immediately
before training, gathers memory measurements from every rank, and writes
`benchmark_summary.json`. It does not alter the production YAML defaults.

`tests/test_sft_jobs.py` includes a deterministic-selector coverage test. These
changes and this report are intentionally uncommitted pending review.

## Verification

- `uv run --with-requirements training/sft/requirements-mlspace.txt pytest -q tests/test_*sft*.py`:
  155 passed, with two pre-existing multiprocessing/fork deprecation warnings.
- `python -m py_compile training/sft/train.py tests/test_sft_jobs.py`: passed.
- `git diff --check`: passed.
- Final `nvidia-smi`: all eight GPUs at 0 MiB and 0% utilization; no training
  or `torchrun` process remained.
