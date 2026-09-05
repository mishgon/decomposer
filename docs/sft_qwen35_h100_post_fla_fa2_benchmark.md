# Qwen3.5-4B SFT FlashAttention-2 benchmark after FLA

> Historical decision: the later representative grouped-batch benchmark
> supersedes this report's batch-1 recommendation. See
> `docs/sft_qwen35_h100_grouped_batch_toolathlon_only_benchmark.md`.

Date: 2026-08-31

## Decision

Keep the current four-H100 production default at per-device batch 1, global
batch 4, with the FLA + causal-conv1d fast path and PyTorch SDPA for the eight
full-attention layers. Do not add locally compiled `flash-attn` as a required
dependency.

On the representative 192-record workload, three rotated measurements per
backend found no native FlashAttention-2 benefit:

- SDPA averaged 97.74 seconds and 12,288 useful tokens/s;
- native `flash-attn==2.8.3` averaged 97.82 seconds and 12,278 tokens/s, a
  0.08% throughput difference from SDPA; and
- the Hugging Face `kernels-community/flash-attn2` backend averaged 100.90
  seconds and 11,902 tokens/s, 3.14% slower than SDPA.

All nine representative runs had exactly the same worst-rank peak memory:
39.64 GiB allocated and 47.57 GiB reserved. The native extension took 38
minutes 41 seconds to compile and produced a 207.55 MiB shared object. That
build and dependency cost has no payoff for the current batch-1 workload.

The focused rerun was still worthwhile because it revealed a different
long-context result. On the longest 24 examples, with batch 2 per GPU:

- SDPA processed 696,058 input tokens in 96.34 seconds;
- the HF FA2 kernel took 36.22 seconds, a 2.66x throughput gain; and
- native FA2 took 40.78 seconds, a 2.36x throughput gain.

Both FA2 implementations reduced worst-rank peak allocation from 71.05 to
66.07 GiB, a 4.983 GiB saving, and reserved memory from 76.92 to 73.93 GiB.
This did not clear the predeclared strict 5.0 GiB gate for trying batch 3. The
previous batch-3 failure needed another 5.18 GiB allocation, so batch 3 was not
attempted without adequate margin.

The practical conclusion is:

- for the current representative batch-1 training pipeline, retain SDPA;
- do not ship the native FA2 build;
- if a future workload is dominated by near-32K records or batch 2 is
  reconsidered, benchmark the HF Hub FA2 kernel on the complete grouped
  workload; it is the more promising FA2 implementation and requires no local
  CUDA build; and
- optimize shared tokenization/validation before treating a few-percent
  attention-kernel difference as an end-to-end job improvement.

These are experimental findings only. No production dependency, YAML, or
project virtual environment was changed, and the benchmark changes remain
uncommitted.

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
- Dataset: teacher-prompt mixed SFT v3, with 1,314 train and 127 validation
  records after runtime checks. Dataset fingerprint:
  `62d42708828c3dca8cde7dbc2bdfb6fd6a7eecf62050c4b126e00cfb72e62308`.
- Representative selection: deterministic environment-proportional and
  length-stratified sample of 192 records, containing 1,200,953 input tokens.
  Lengths span 2,765-30,149 tokens, with median 3,849, p90 14,168, p95 18,874.
- Representative data manifest SHA-256:
  `80d032d5d012fde8b87ee4d807425aeba2898becd2f50506841eb43a13a6c1a6`.
  Every backend and repeat used that same manifest.
- Capacity selection: the longest 24 records, containing 656,073 useful
  unpadded tokens and 696,058 input tokens as counted by the trainer. Lengths
  span 23,521-32,626 tokens, with median 26,901, p90 31,411, and p95 32,008.
- Hardware: four of eight otherwise-idle NVIDIA H100 80 GB HBM3 GPUs, driver
  570.133.20. Every GPU run explicitly used
  `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Base runtime: Python 3.12.13, PyTorch 2.11.0+cu129, Transformers 5.14.1,
  TRL 1.9.2, and Liger Kernel 0.8.1.
- Linear-attention fast path: `fla-core==0.5.2`,
  `flash-linear-attention==0.5.2`, `causal-conv1d==1.7.0`, and an isolated
  `triton==3.7.1` overlay, with `FLA_TILELANG=0`.
- HF backend: `kernels==0.15.2`, `kernels-data==0.16.1`, and Transformers'
  default version 1 of `kernels-community/flash-attn2`.
- Native backend: source-built `flash-attn==2.8.3`, compiled for SM90 with an
  isolated CUDA 12.9.86 compiler, `MAX_JOBS=4`, and `NVCC_THREADS=2`.
- Raw artifacts:
  `/home/sukhorukov/decomposer_artifacts/training/sft/benchmarks/qwen35-4b-h100-post-fla-fa2-20260830`.

The new artifact root is 219 MiB, of which 211 MiB is the native overlay. The
native CUDA shared object is 217,636,792 bytes with SHA-256
`d47f39096a5c8b7dedb308a83cc1e27b4e49fae8bd0ed1fd4b916715c6cbdac0`.
It reuses the isolated compiler, FLA, causal-conv1d, Triton, and populated
kernel-cache artifacts documented in
`docs/sft_qwen35_h100_linear_attention_benchmark.md`.

## Benchmark design

The three backends were:

| Label | Attention setting | Resolved full-attention implementation |
| --- | --- | --- |
| SDPA | `sdpa` | `transformers.integrations.sdpa_attention.sdpa_attention_forward` |
| HF FA2 | `kernels-community/flash-attn2` | Hub `_flash_attn2_cuda_f12afc9` function through the Transformers FA2 wrapper |
| Native FA2 | `flash_attention_2` | `flash_attn.flash_attn_interface.flash_attn_func` through the same wrapper |

The benchmark summary records package versions, callable identities, source or
extension paths, sizes, and SHA-256 hashes. This is necessary because the
common Transformers wrapper does not by itself distinguish the HF Hub kernel
from the native package.

Three warmups were excluded from the measured aggregate. The nine batch-1
runs used this rotated order:

1. SDPA, HF FA2, native FA2;
2. HF FA2, native FA2, SDPA; and
3. native FA2, SDPA, HF FA2.

The common representative command shape was:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
FLA_TILELANG=0 \
TRITON_CACHE_DIR=/absolute/populated/cache \
PYTHONPATH=/absolute/triton371:/absolute/causal:/absolute/fla:$PWD \
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

HF runs changed the attention implementation to
`kernels-community/flash-attn2`. Native runs used `flash_attention_2` and
prepended the isolated native overlay to `PYTHONPATH`. Capacity runs changed
the per-device/global batch to 2/8 and used `--longest-train-samples 24`.

## Numerical smoke test

A deterministic BF16 forward/backward test used the actual Qwen attention
dimensions: query shape `[2, 257, 16, 256]`, key/value shape
`[2, 257, 4, 256]`, and causal attention. Outputs and Q/K/V gradients were
finite for every backend.

| Comparison | Output relative L2 | Output cosine | Q-grad cosine | K-grad cosine | V-grad cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF FA2 vs SDPA | 0.000410 | 0.99999988 | 0.99999994 | 0.99999952 | 0.99999988 |
| Native FA2 vs SDPA | 0.000410 | 0.99999988 | 0.99999988 | 0.99999958 | 1.00000000 |
| Native FA2 vs HF FA2 | 0.000019 | 1.00000012 | 1.00000000 | 0.99999994 | 1.00000000 |

The small BF16 differences are consistent with different reduction orders and
do not show gross forward or backward divergence.

## Representative batch-1 results

Every row trained the same 192 records for 48 optimizer steps. Throughput is
the 1,200,953 unpadded input-token count divided by trainer runtime. Standard
deviation is the sample standard deviation across the three rotated runs.

| Backend | R1 / R2 / R3 runtime | Mean +/- stdev | Mean useful tokens/s | Change vs SDPA | Worst allocated / reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| SDPA | 97.16 / 100.70 / 95.34 s | 97.74 +/- 2.72 s | 12,288 | baseline | 39.64 / 47.57 GiB |
| HF FA2 | 99.83 / 98.90 / 103.98 s | 100.90 +/- 2.70 s | 11,902 | -3.14% | 39.64 / 47.57 GiB |
| Native FA2 | 100.68 / 99.14 / 93.63 s | 97.82 +/- 3.71 s | 12,278 | -0.08% | 39.64 / 47.57 GiB |

Native FA2 is statistically and operationally tied with SDPA. HF FA2 is
slower on this representative batch-1 workload. The identical memory values
also show that neither FA2 implementation creates additional batch-1 capacity.

## Longest-context batch-2 results

Every backend trained the same longest 24 records for three optimizer steps.
This is a memory/capacity stress test and is not representative of the full
dataset's length distribution.

| Backend | Result | Runtime | Input tokens/s | Speed vs SDPA | Worst allocated / reserved |
| --- | --- | ---: | ---: | ---: | ---: |
| SDPA | passed | 96.34 s | 7,225 | baseline | 71.05 / 76.92 GiB |
| HF FA2 | passed | 36.22 s | 19,216 | 2.66x | 66.07 / 73.93 GiB |
| Native FA2 | passed | 40.78 s | 17,070 | 2.36x | 66.07 / 73.93 GiB |

FA2 matters when the full-attention layers see very long, two-example batches.
The HF kernel was 12.57% faster than the native package on this exact capacity
test while using identical peak memory. That result favors the HF backend if
the workload later shifts toward long-context batch 2.

The strict batch-3 gate used worst-rank allocated memory:

```text
71.0503 GiB SDPA - 66.0670 GiB FA2 = 4.9833 GiB saved
```

Because 4.9833 GiB is below the predeclared 5.0 GiB threshold and the earlier
batch-3 OOM requested another 5.18 GiB, no batch-3 attempt was made.

## Problems encountered and resolutions

1. The project environment still has no native `flash-attn` package. The
   isolated CUDA 12.9 compiler prepared for the FLA experiment made a focused
   source build possible without modifying `.venv` or requirements.
2. `flash-attn==2.8.3` compiled many SM90 CUDA translation units and took 38
   minutes 41 seconds even with `MAX_JOBS=4` and `NVCC_THREADS=2`. The result is
   a 207.55 MiB extension, making native installation materially more costly
   than the HF Hub kernel.
3. The initial provenance helper looked for the FlashAttention wrapper in the
   wrong Transformers module. In Transformers 5.14.1 the wrapper is in
   `transformers.integrations.flash_attention`, while lazy kernel loading is in
   `modeling_flash_attention_utils`. The helper now imports each from its
   actual module.
4. Querying Transformers' attention registry before lazy-loading the Hub
   backend reports only the generic wrapper. Loading the requested backend
   first, then recording both the wrapper and the returned functions, captures
   the actual HF or native implementation.
5. Transformers selected the supported default version 1 of the HF Hub kernel
   and warned that version 3 exists. The default was retained to avoid adding
   an unpinned variable to the comparison.
6. Every distributed rank repeats full prepared-data validation and launches
   `num_proc=8` dataset maps before TRL tokenizes again. With four ranks this
   can spawn 32 tokenizer workers per map. The work is outside
   `trainer.train()` timing, but it makes benchmark and real-job startup longer
   than a warm batch-1 epoch. Sharing validated/tokenized data remains a higher
   priority end-to-end optimization.
7. The batch-1 result and batch-2 longest-context result differ sharply because
   Qwen3.5-4B has only eight full-attention layers and the representative data
   contains many much shorter examples. FA2's quadratic-attention advantage is
   visible only when those eight layers process the longest, larger batches.
8. `torchrun` emitted IPv6 address-family warnings, but every distributed run
   initialized, completed, and released GPU memory normally.
9. `rg` and `jq` are absent on this host. Inspection used `grep`, `sed`, and
   small read-only Python aggregations; benchmark behavior was unaffected.

## Uncommitted benchmark support

`training/sft/train.py` contains the still-uncommitted benchmark CLI and
runtime helpers from this and the preceding batch/FLA experiments. This rerun
adds exact full-attention backend provenance to `benchmark_summary.json`,
including package versions, callable identities, file sizes, and hashes.

`tests/test_sft_jobs.py` checks SDPA callable provenance. The native and Hub
implementations were additionally verified from every full benchmark summary.
No production configuration or dependency file was changed.

## Verification

- `CUDA_VISIBLE_DEVICES='' uv run --with-requirements
  training/sft/requirements-mlspace.txt pytest -q tests/test_*sft*.py`: 159
  passed, with two pre-existing multiprocessing/fork deprecation warnings.
- `python -m py_compile training/sft/train.py tests/test_sft_jobs.py`: passed.
- `git diff --check`: passed.
- All 15 GPU training runs, the numerical smoke test, and the native build
  completed successfully. Batch 3 was intentionally skipped by the memory
  gate, not failed.
- Final `nvidia-smi`: all eight GPUs at 0 MiB and 0% utilization; no training
  or `torchrun` process remained.
