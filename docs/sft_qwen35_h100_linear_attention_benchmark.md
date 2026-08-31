# Qwen3.5-4B SFT linear-attention kernel benchmark

> Historical decision: the later grouped batch-2 HF FA2 benchmark supersedes
> this report's batch-1 production recommendation. See
> `docs/sft_qwen35_h100_grouped_batch_toolathlon_only_benchmark.md`.

Date: 2026-08-30

## Decision

The Qwen3.5 linear-attention fast path is worth integrating. On the same
representative 192-record, 1,200,953-token workload, the combined
`flash-linear-attention` and `causal-conv1d` path:

- reduced warm training time from 761.45 to 98.26 seconds;
- increased useful throughput from 1,577 to 12,222 tokens/s (7.75x);
- reduced worst-rank peak allocated memory from 51.57 to 39.64 GiB;
- reduced worst-rank peak reserved memory from 59.89 to 47.57 GiB; and
- remained finite and numerically close to the BF16 Torch reference in a
  forward/backward layer smoke test.

`flash-linear-attention` provides almost all of the improvement.
`causal-conv1d` alone improved throughput by only 1.45% and saved 0.46 GiB,
but adding it to FLA improved the comparable cold run by another 4.56% and
enabled Transformers' complete fast-path identity. It is reasonable to ship
both as a tested pair.

Do not raise the production microbatch yet. Batch 2 per GPU now fits the
longest 32K records, while batch 3 OOMs, but batch 2 is slower on this
variable-length dataset:

- random batching adds 36.38% padding and reduces useful warm throughput to
  5,333 tokens/s;
- length grouping reduces padding to 3.80%, but two stable runs still reach
  only 9,373 and 9,314 useful tokens/s, 23-24% below batch 1.

The current production recommendation is therefore per-device batch 1,
global batch 4, with the linear-attention fast path. Batch 2/global 8 is the
verified memory ceiling and can be reconsidered if a later sampler, packing
scheme, or model/runtime version makes it faster.

These are experimental findings only. No dependency or production config was
committed.

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
- Dataset: the teacher-prompt mixed SFT v3 dataset used by the config. It has
  1,314 train and 127 validation records after runtime checks. Valid train
  lengths range from 2,734 to 32,626 tokens (median 3,872, p90 14,007, p95
  18,837).
- Representative sample: deterministic environment-proportional,
  length-stratified selection of 192 records. It contains 1,200,953 useful
  tokens and spans 2,765-30,149 tokens (median 3,849, p90 14,168, p95 18,874).
- Capacity sample: the longest 24 records for batch 2 and longest 36 records
  for batch 3, giving three optimizer-step attempts in each case. The batch-2
  sample spans 23,521-32,626 tokens.
- Hardware: four of eight otherwise-idle NVIDIA H100 80 GB HBM3 GPUs, driver
  570.133.20. Every run explicitly used `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Base runtime: Python 3.12.13, PyTorch 2.11.0+cu129, Transformers 5.14.1,
  TRL 1.9.2, and Liger Kernel 0.8.1.
- Fast-path packages: `fla-core==0.5.2`,
  `flash-linear-attention==0.5.2`, source-built
  `causal-conv1d==1.7.0`, and an isolated `triton==3.7.1` overlay.
- CUDA compiler overlay: `cuda-nvcc==12.9.86` in an isolated 1.3 GB conda
  prefix. The project virtual environment was not replaced.
- Backend policy: `FLA_TILELANG=0`, forcing FLA's Triton backend. The installed
  TileLang path is not usable in this environment; details are below.
- Raw artifacts:
  `/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/training/sft/benchmarks/qwen35-4b-h100-linear-attn-20260830`.

Artifact sizes at the end of the experiment were 3.4 GB total: 1.3 GB CUDA
compiler prefix, 694 MB Triton overlay, 283 MB causal-conv1d overlay, 13 MB
FLA overlay, and 604 MB of populated combined Triton kernel cache. A production
image should not reproduce this ad hoc layout; it should build a pinned wheel
or image layer and persist/prewarm the kernel cache.

Selected artifact hashes:

- causal CUDA extension:
  `2345460f9ca048060b81f1287f381b6dcf8c1e7ca74c9ae3488729a6f5392847`;
- FLA gated-delta chunk implementation:
  `fd4e01dc22a8c139c2a6eb61e47ae472a50322e4b4fff006cc5039a4602b310e`;
- Triton 3.7.1 module entry point:
  `fc8d0ce7535a4773769dfdcdb3a8b7bcee1bf58d3d03cc318529104c1e1431fd`.

## Commands

The representative command shape was:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
FLA_TILELANG=0 \
TRITON_CACHE_DIR=/absolute/artifact/cache \
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

The four ablations changed `PYTHONPATH` only:

| Variant | Optional overlays | Transformers complete fast-path flag |
| --- | --- | --- |
| Torch fallback | none | false |
| causal only | causal-conv1d | false |
| FLA only | Triton 3.7.1 + FLA | false |
| combined | Triton 3.7.1 + causal-conv1d + FLA | true |

Although the two single-package controls do not set the complete flag, runtime
provenance and the layer smoke verify that their respective optional callables
are bound and executed. The benchmark summary records package versions and the
fully qualified identities of causal convolution, gated-delta, recurrent, and
fused normalization implementations.

Batch-2 grouping used the Transformers 5.14 API
`train_sampling_strategy="group_by_length"` and the prepared dataset's exact
`_token_length` column. The experimental CLI exposes this as
`--group-by-length`.

## Numerical smoke test

An actual `Qwen3_5GatedDeltaNet` layer from the pinned model configuration was
run in BF16 with batch 2, sequence length 257, a padded attention mask, and a
deterministic forward/backward loss. Every variant produced finite outputs,
input gradients, and parameter gradients.

| Variant | Output relative L2 | Output cosine | Input-grad relative L2 | Input-grad cosine | Peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| causal only | 0.00690 | 0.999976 | 0.01009 | 0.999949 | 513.6 MiB |
| FLA only | 0.00602 | 0.999982 | 0.00925 | 0.999957 | 523.7 MiB |
| combined | 0.00781 | 0.999970 | 0.01156 | 0.999933 | 515.6 MiB |

The maximum output difference was 0.00390625 for each optional path. The high
cosine similarity and small relative error are consistent with a different
BF16 reduction order. They do not show gross forward or gradient divergence.
The FLA cold smoke spent about 104 seconds compiling; that time is not a useful
single-layer throughput number.

## Representative results

All batch-1 rows trained the exact same 192 records for 48 optimizer steps.
"Useful tokens/s" uses the unpadded token total from the prepared metadata.

| Linear-attention path | Cache state | Runtime | Useful tokens/s | Change vs Torch | Worst allocated / reserved |
| --- | --- | ---: | ---: | ---: | ---: |
| Torch fallback | n/a | 761.45 s | 1,577 | baseline | 51.57 / 59.89 GiB |
| causal only | n/a | 750.57 s | 1,600 | +1.45% | 51.11 / 59.42 GiB |
| FLA only | cold | 499.36 s | 2,405 | +52.49% | 40.10 / 47.92 GiB |
| FLA + causal | cold | 477.61 s | 2,515 | +59.43% | 39.64 / 47.57 GiB |
| FLA + causal | warm | 98.26 s | 12,222 | +674.95% (7.75x) | 39.64 / 47.57 GiB |

The combined cold run is 4.56% faster than FLA-only cold. Both used separate
empty caches, so each paid the shape-compilation cost. The warm combined run
reused the populated cache in a fresh `torchrun` process and demonstrates that
the disk cache is effective across jobs.

Cold compilation is operationally significant. The first combined optimizer
step took 124.6 seconds cold and 18.5 seconds warm. Later unseen sequence/batch
shapes also caused alternating slow compilation steps and fast steady-state
steps. The complete representative batch-1 cache occupied hundreds of MB, and
adding batch-2 shapes grew the shared cache to 604 MB.

## Batch capacity and batching results

| Per-GPU / global batch | Sampler/workload | Result | Runtime | Useful tokens/s | Padding | Worst allocated / reserved |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 / 4 | representative, random | passed | 98.26 s | 12,222 | 0.00% | 39.64 / 47.57 GiB |
| 2 / 8 | longest 24 | 3 steps passed | 180.07 s | 3,643 | 6.09% | 71.05 / 76.92 GiB |
| 3 / 12 | longest 36 | OOM in first step | n/a | n/a | n/a | 73.44 GiB allocated before failure |
| 2 / 8 | representative, random, cache fill | passed | 412.12 s | 2,914 | 36.38% | 66.24 / 74.33 GiB |
| 2 / 8 | representative, random, warm | passed | 225.19 s | 5,333 | 36.38% | 66.24 / 74.33 GiB |
| 2 / 8 | representative, grouped | passed | 128.14 s | 9,373 | 3.80% | 60.46 / 68.41 GiB |
| 2 / 8 | representative, grouped repeat | passed | 128.94 s | 9,314 | 3.80% | 60.46 / 68.41 GiB |

Batch 3 failed inside a full-attention SDPA layer, not FLA: PyTorch tried to
allocate another 5.18 GiB with 4.05 GiB free and 73.44 GiB already allocated.
This establishes batch 2 as the worst-case full-32K ceiling. Batch 2's longest
pass had only about 2.7 GiB of physical memory headroom at peak, so it should
not be treated as a generous margin.

The random batch-2 cache-fill and warm rows show that two issues are distinct:
JIT specialization accounts for the 412-to-225-second reduction, while random
padding remains after warmup. Length grouping fixes padding, but not enough to
beat batch 1. The two grouped measurements are effectively identical, so the
remaining difference is not residual compilation noise.

## Problems encountered and resolutions

1. The base virtual environment had no usable `nvcc` for this extension build.
   An isolated CUDA 12.9.86 compiler prefix was installed under the benchmark
   artifact root. The base environment and project requirements were not
   modified.
2. No upstream causal-conv1d wheel matched Python 3.12, PyTorch 2.11, and this
   CUDA runtime, so `causal-conv1d==1.7.0` required a source build. The source
   build also compiles several GPU architectures and is slow/noisy.
3. The first causal-conv1d build compiled CUDA objects but failed its C++
   object with `cuda_runtime_api.h: No such file or directory`. Conda places
   CUDA headers under `targets/x86_64-linux/include`, while Torch's extension
   helper searched `$CUDA_HOME/include`. Setting `CPATH` to the target include
   directory and `LIBRARY_PATH`/`LD_LIBRARY_PATH` to the target library
   directory, using `/usr/bin/gcc` and `/usr/bin/g++`, and limiting
   `MAX_JOBS=4` completed the build while reusing existing objects.
4. FLA initially auto-selected the already installed TileLang 0.1.9 backend.
   TileLang selected a pip CUDA 13 `nvcc` path that lacked the `cuda/atomic`
   headers and failed with `fatal error: cuda/atomic: No such file or
   directory`. Setting `FLA_TILELANG=0` selected the supported Triton backend.
   This environment variable is mandatory for this tested environment.
5. The base Triton is 3.6.0. FLA 0.5.2 intentionally rejects gated
   `chunk_bwd_dqkwg` on Hopper with Triton >=3.4.0 and <3.7.1 because that range
   can produce incorrect results. The benchmark did not bypass the guard. An
   isolated `triton==3.7.1` overlay made the backward smoke and full training
   pass.
6. Installing Triton 3.7.1 briefly started two download processes against the
   same artifact target after the first shell call outlived an early output
   yield. The duplicate process was terminated by exact PID; the attached
   installation then completed. No GPU or project process was affected.
7. GNU `/usr/bin/time` is absent on this image. The first attempted command
   exited before Python/GPU allocation. The benchmark harness's synchronized
   per-rank timers and CUDA peak counters were used instead.
8. Every `torchrun` rank independently performs the full prepared-data
   validation and then launches `num_proc=8` tokenization maps before TRL
   tokenizes training input. A four-GPU invocation can therefore create 32
   tokenizer workers per map. This startup cost is outside the timed
   `trainer.train()` interval but is now longer than the 98-second warm epoch.
   It should be fixed independently by validating/tokenizing once and sharing
   a reusable cache.
9. The first length-grouping attempt used the old `group_by_length` training
   argument and failed before model construction. Transformers 5.14.1 renamed
   this to `train_sampling_strategy="group_by_length"`. Using the current API
   with `length_column_name="_token_length"` resolved it.
10. `torchrun` prints IPv6 address-family warnings on startup. Distributed
    training nevertheless initializes and releases all ranks correctly.
11. `rg` and `jq` are absent on this host; log inspection used `grep`, `sed`,
    and small read-only Python summaries.

## Uncommitted benchmark support

`training/sft/train.py` currently contains uncommitted benchmark helpers from
this and the preceding FlashAttention-2 experiment:

- batch, attention-backend, longest/stratified selection, and length-grouping
  CLI overrides;
- a benchmark mode that disables evaluation, checkpoints, ClearML, and final
  export;
- synchronized per-rank training time and CUDA memory collection;
- deterministic environment/length-stratified sampling; and
- Qwen3.5 linear-attention package/callable provenance in
  `benchmark_summary.json`.

`tests/test_sft_jobs.py` covers the deterministic selector and the runtime
provenance schema. The production YAML remains unchanged, and none of these
changes have been committed.

## Verification

- `uv run --with-requirements training/sft/requirements-mlspace.txt pytest -q
  tests/test_*sft*.py`: 158 passed, with two pre-existing multiprocessing/fork
  deprecation warnings.
- The targeted selector, runtime-provenance, and Transformers 5.14 sampler API
  tests: 4 passed.
- `python -m py_compile training/sft/train.py tests/test_sft_jobs.py`: passed.
- `git diff --check`: passed.
- Final `nvidia-smi`: all eight GPUs at 0 MiB and 0% utilization; no training
  or `torchrun` process remained.

All raw successful summaries, expected-failure logs, build logs, package
overlays, numerical tensors, and cache directories remain under the artifact
root above.
