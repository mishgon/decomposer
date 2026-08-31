# Qwen3.5 H100 fast SFT runtime

The production Qwen3.5-4B mixed-v3 recipe uses two independent accelerators:

- `flash-linear-attention` and `causal-conv1d` accelerate the model's 24
  gated-delta linear-attention layers;
- the pinned Hugging Face Hub FlashAttention-2 kernel accelerates its eight
  full-attention layers.

The selected configuration is per-device batch 2 on four H100s (global batch
8), with examples grouped by their prepared `_token_length`. The benchmark and
rationale are in
`docs/sft_qwen35_h100_grouped_batch_toolathlon_only_benchmark.md`.

## What needs NVCC

HF FlashAttention-2 does **not** need NVCC. It is delivered by the `kernels`
client as a precompiled Hub kernel. FLA uses Triton JIT and also does not need NVCC during a
training job.

Only `causal-conv1d==1.7.0` needs a one-time native build because there is no
matching upstream wheel for Python 3.12, Torch 2.11, and CUDA 12.9. The result
is stored in a reusable runtime bundle. MLSpace and local training jobs consume
that bundle and do not need a CUDA compiler.

## One-time preparation

Create a pinned CUDA 12.9 compiler prefix. The compiler is an environment
preparation artifact, not a project dependency:

```bash
/home/user/conda/bin/conda create -y --override-channels \
  -c nvidia -c conda-forge \
  -p /home/jovyan/decomposer-artifacts/toolchains/cuda-12.9.86 \
  cuda-nvcc=12.9.86
```

Synchronize the checked-in training dependencies, then build the immutable
H100 bundle. The build compiles only SM90 code and uses at most four compiler
workers:

```bash
uv sync --locked --group train

.venv/bin/python -m training.sft.prepare_qwen35_fast_runtime \
  --cuda-home /home/jovyan/decomposer-artifacts/toolchains/cuda-12.9.86 \
  --hf-cache-dir /mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface \
  --max-jobs 4
```

The default output is:

```text
/home/jovyan/decomposer-artifacts/kernels/sft/qwen35-hf-fa2-fla-v1
```

It contains isolated Triton 3.7.1 and causal-conv1d overlays plus a manifest
with ABI, package identity, sizes, and SHA-256 checksums. The project venv keeps
Torch's locked Triton 3.6 dependency; the launcher prepends the validated 3.7.1
overlay because FLA rejects Triton 3.4-3.6 on H100 for correctness.

Verify an existing bundle without rebuilding it:

```bash
.venv/bin/python -m training.sft.prepare_qwen35_fast_runtime --verify-only
```

Bundle creation refuses to overwrite an existing directory. A version change
must use a new runtime profile and a new bundle path.

## Local smoke

Normal runs must set `FLA_TILELANG=0`; the installed TileLang backend is not the
validated path. Each experiment also gets its own writable Triton cache. The
first unseen batch/sequence shapes can spend several minutes compiling, while
later epochs reuse the disk cache.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FLA_TILELANG=0 \
DECOMPOSER_SFT_RUNTIME_PROFILE=qwen35-hf-fa2-fla-v1 \
DECOMPOSER_SFT_RUNTIME_BUNDLE=/home/jovyan/decomposer-artifacts/kernels/sft/qwen35-hf-fa2-fla-v1 \
TRITON_CACHE_DIR=/home/jovyan/decomposer-artifacts/cache/triton/sft/local-qwen35-fast-smoke \
PYTHONPATH=/home/jovyan/decomposer-artifacts/kernels/sft/qwen35-hf-fa2-fla-v1/triton:/home/jovyan/decomposer-artifacts/kernels/sft/qwen35-hf-fa2-fla-v1/causal:$PWD/src:$PWD \
.venv/bin/torchrun --standalone --nproc-per-node=4 \
  -m training.sft.train \
  --config training/sft/configs/qwen35_4b_nonthinking_mixed_v3_gaia2_execution_110_n7_teacher_prompt_filtered_32k_hf_fa2_fla_b8_smoke_4gpu.yaml \
  --output-dir /home/jovyan/decomposer-artifacts/training/sft/local-qwen35-fast-smoke \
  --benchmark
```

The trainer fails before loading the dataset if the bundle, H100 capability,
package versions, FLA/causal callables, or pinned HF FA2 callable do not match.
The resolved runtime provenance is saved in `resolved_config.json` and the
smoke result in `benchmark_summary.json`.

## MLSpace jobs

The launcher creates/reuses a training venv keyed by `uv.lock`, verifies the
runtime bundle on the submitter, stages the clean Git commit, and injects all
runtime environment variables. NVCC is not used by a job.

Dry-run the two exact payloads:

```bash
/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m training.sft.run_train_jobs --dry --sanity-check --priority high \
  --filter hf-fa2-fla-b8-smoke-4gpu

/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m training.sft.run_train_jobs --dry --priority high \
  --filter hf-fa2-fla-b8-full-4gpu
```

Remove `--dry` to submit. Smoke artifacts and logs are under
`/home/jovyan/decomposer-artifacts/training/sft/jobs_sanity/<experiment>/`;
full-run artifacts are under the corresponding `jobs/<experiment>/` path.
`console.log` is the primary combined log.

`--force` starts a fresh run. Any existing non-empty output is first moved
atomically to `jobs/_archive/<experiment>/<UTC timestamp>-<commit>`. With
`--force --resume-latest`, the current output stays in place and the latest
complete checkpoint is resumed.

## Operational notes

- Do not install native `flash-attn`; the HF kernel was faster for batch 2 and
  avoids a roughly 39-minute source build.
- Do not use batch 3. It OOMed on the first longest-context backward pass and
  had no production memory margin.
- The four distributed ranks still repeat preflight tokenization. Removing
  that startup work is intentionally deferred to a separate change because it
  affects dataset-validation and cache coordination rather than GPU kernels.
- Changing global batch 4 to 8 halves optimizer updates at the same five
  epochs. The recipe deliberately keeps the old `1e-5` learning rate and token
  exposure; downstream GAIA2 and Workplace evaluation must measure quality.
