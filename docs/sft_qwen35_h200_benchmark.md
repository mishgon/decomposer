# Qwen3.5-4B SFT on H200: Benchmark and Step Profile

Date: 2026-10-07

## Decision

On Hertz-2's H200 GPUs, train the Qwen3.5-4B unloop student with plain DDP. The model's weights stay in fp32, compute runs in bf16 under autocast, and activation checkpointing stays on. Use per-device batch 2 and global batch 8.

On the release described below, this reached 22,108 useful tokens/s on four H200s. That is 15% more than the FSDP recipe used so far (19,234 tokens/s), and its loss trajectory matches FSDP's. Batch size and FSDP re-sharding make no measurable difference. Without activation checkpointing, even batch 1 runs out of memory on the 32K records.

Three environment settings are required for any Qwen3.5 SFT job on Hertz-2 (see **Environment**). Without them, jobs fail, or lose half their throughput to kernel autotuning.

A one-step profile shows where the remaining time goes. At the median length, a single GPU spends:

| Share of GPU time | Where |
| ---: | --- |
| 39% | GEMMs |
| 26% | Memory copies and casts inside backward and recomputation |
| 7% | Unfused elementwise work |
| 7% | FlashAttention, rising to 24% for 28K-token records |
| 7% | FLA gated-delta kernels |
| 6% | PyTorch's fallback for the linear-attention short convolution |
| 5% | The optimizer |

DDP itself costs about 4% when every rank gets the same lengths. **Opportunities** ranks what could still be gained.

## Setup

- **Hardware and data:**
  - 4× NVIDIA H200 NVL, 141 GB each (GPUs 4-7 of Hertz-2), driver 580.178.04.
  - Model: the unloop checkpoint (Qwen3.5-4B plus a merged DPO LoRA): `/mnt/share14T-2/sukhorukov/decomposer_artifacts/models/qwen35_4b_original_unloop/checkpoint-0`.
  - Release: `decomposer-mixed-qwen38-qwen35-4b-unloop-nonthinking/v1-tau2-broad-workplace-train-n1-32k`, fingerprint `d004e292…dfc3060`.
- **Software:**
  - Torch 2.11.0+cu129, Transformers 5.14.1, TRL 1.9.2, accelerate 1.14.0.
  - flash-linear-attention 0.5.2, liger-kernel 0.8.1, kernels 0.15.2.
  - Triton 3.7.1 from the overlay. `causal-conv1d` is **not** installed.
  - Decomposer at `b1a6726` (`feat/sft_lora`).
- **Model structure:** 32 layers, 24 of them gated-delta linear attention and 8 full attention (16 query heads, 4 KV heads, head dim 256). MLP width 9,216, vocabulary 248,320. The checkpoint is the multimodal `Qwen3_5ForConditionalGeneration`, so it also carries a vision tower and an MTP head that text SFT never uses.
- **Training recipe** (from the existing H100 one):
  - learning rate 1e-5, cosine schedule, fused AdamW, weight decay 0.1;
  - Liger fused linear cross-entropy; the profile confirmed the model's `forward` comes from `liger_kernel.transformers.model.qwen3_5`;
  - the pinned HF FA2 kernel (`kernels-community/flash-attn2@c269cc5…`) for the 8 attention layers, and FLA for the 24 linear-attention layers;
  - length-grouped batches, assistant-only loss.
- **Sample:** the trainer's deterministic environment-and-length-stratified sample of 256 training records.
  - 2,148,552 tokens, 426,902 of them supervised;
  - lengths 4,321 to 27,166 (p50 7,280, p90 13,006);
  - ordered-record checksum `1bdc4dd8bf1cbdc48ceb538c52dc55a91a4056b57a33c6fcec6d4866b30804ec`, identical in every timed case.
- **Memory gates:** before each timed case, a gate ran the 4 × batch × 2 longest records, i.e. the longest batch on every rank.
- **Measurement:**
  - Throughput is useful (unpadded) tokens divided by the slowest rank's trainer time for one epoch over the sample.
  - Each configuration was measured warm, i.e. after its kernel autotuning had been cached (see **Problems**).
  - The driver sampled the processes on GPUs 4-7 every 10 s. No timed case below shared a GPU with another job.

## Benchmark Results

All cases use 4 GPUs and global batch = 4 × per-device batch.

| Configuration | Batch / GPU | Useful tokens/s | Per GPU | Padding | Peak allocated | Train loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FSDP, current recipe (full shard, re-shard after forward, activation checkpointing) | 2 | 19,234 | 4,808 | 1.8% | 50.9 GiB | 0.5015 |
| FSDP, batch 3 | 3 | 17,035 | 4,259 | 9.7% | 79.5 GiB | 0.5122 |
| FSDP, batch 4 | 4 | 19,230 | 4,808 | 5.2% | 92.4 GiB | 0.5335 |
| FSDP, `reshard_after_forward: false` | 2 | 19,598 | 4,899 | 1.8% | 56.9 GiB | 0.5021 |
| DDP, bf16 weights | 2 | 22,375 | 5,594 | 1.8% | 50.5 GiB | 0.5116 |
| **DDP, fp32 weights, bf16 autocast** | **2** | **22,108** | **5,527** | **1.8%** | **91.7 GiB** | **0.5012** |
| No activation checkpointing (FSDP or DDP) | 1 | out of memory on the longest records | | | | |

**Memory on the longest records:**

| Configuration | Peak allocated |
| --- | ---: |
| FSDP batch 3 | 83.8 GiB |
| FSDP batch 4 | 106.7 GiB |
| DDP bf16 batch 2 | 56.4 GiB |
| DDP fp32 batch 2 | 100.0 GiB |

The training losses are not a quality comparison: a different global batch changes the number and composition of optimizer steps.

**Batch size and re-sharding.** Batch 4 runs exactly as fast as batch 2 and needs almost twice the memory. Batch 3 is slower because length grouping pairs records less evenly, which leaves more padding. Skipping FSDP re-sharding is within noise. The step time is spent inside the layers, not on batching or communication.

**Why bf16 DDP is excluded.** accelerate upcasts trainable bf16 weights to fp32 only on its FSDP path (`fsdp2_prepare_model`). Plain DDP with a bf16 checkpoint therefore trains entirely in bf16: weights and Adam state alike. The first step matches FSDP (loss 0.6495 against 0.6489, gradient norm 18.75 against 18.76), but the trajectories drift once updates begin, because many 1e-5-scale updates fall below bf16 resolution. Loading the model in fp32 (`model.dtype: float32`) restores FSDP's trajectory step for step:

| Step | 1 | 5 | 9 | 13 | 17 | 21 | 25 | 29 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FSDP | 0.6489 | 0.6338 | 0.478 | 0.5068 | 0.4126 | 0.4873 | 0.4365 | 0.498 |
| DDP, bf16 weights | 0.6495 | 0.6527 | 0.4565 | 0.5082 | 0.4067 | 0.5007 | 0.4607 | 0.5207 |
| DDP, fp32 weights | 0.6488 | 0.6385 | 0.479 | 0.5073 | 0.4107 | 0.4829 | 0.4348 | 0.4951 |

**DDP settings.** DDP needs `ddp_find_unused_parameters: true`, because the vision tower and MTP head receive no gradients. It also needs non-reentrant gradient checkpointing (`gradient_checkpointing_kwargs: {use_reentrant: false}`). Each rank holds the full fp32 weights, gradients and Adam state, which is why the peak is about 92 GiB at batch 2. Batch 4 would not fit on the longest records.

**Comparison with H100.** The H100 report measured 17,491 tokens/s on four H100s, but on a shorter release (p50 3,849) and with `causal-conv1d` installed. The two numbers are not directly comparable. Per GPU, the H200 FSDP recipe is about 10% faster.

## Step Profile

`profile_step.py` runs the trainer's model path on one GPU:
- TRL's architecture choice;
- Liger's instance patch;
- fp32 weights under bf16 autocast, with non-reentrant checkpointing and fused AdamW;
- batch 2 of real release records, with assistant-only labels.

For each length class it runs 3 warm-up steps, 5 timed steps with CUDA-event timers per phase and per module, and 2 steps under the PyTorch profiler. Each length class ran on its own GPU.

### By Length

| Records near | Real tokens per step | Step time | Tokens/s (1 GPU) | Forward | Backward incl. recompute | Clip + optimizer | Peak allocated |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3K | 7,759 | 1,144 ms | 6,781 | 458 ms | 636 ms | 51 ms | 66.5 GiB |
| 7.5K | 15,002 | 1,842 ms | 8,145 | 619 ms | 1,172 ms | 51 ms | 67.7 GiB |
| 16K | 31,978 | 3,977 ms | 8,041 | 1,205 ms | 2,720 ms | 51 ms | 72.2 GiB |
| 28K | 55,656 | 7,483 ms | 7,438 | 2,101 ms | 5,331 ms | 50 ms | 82.1 GiB |

### GPU Kernel Time by Category

Shares are of total kernel time per step; that total matches wall time to within 2%, so the GPU is never idle.

| Category | 3K | 7.5K | 16K | 28K | What it is |
| --- | ---: | ---: | ---: | ---: | --- |
| GEMM | 42% | 39% | 37% | 33% | cuBLAS Hopper kernels for all linear layers, including the output layer inside Liger's loss |
| Copies and casts | 27% | 26% | 24% | 21% | mostly `aten::copy_` inside backward and recomputation, plus fp32↔bf16 casts |
| Elementwise | 7% | 7% | 7% | 7% | unfused SiLU and multiply (SwiGLU), adds, RMSNorm arithmetic |
| FlashAttention (8 layers) | 3% | 7% | 15% | 24% | HF FA2 forward and backward, head dim 256 |
| FLA gated delta (24 layers) | 6% | 7% | 7% | 7% | chunked delta rule, L2 norm, gated norm, cumsum |
| Short convolution fallback | 6% | 6% | 7% | 6% | PyTorch depthwise `conv1d`, forward and backward |
| Optimizer | 7% | 5% | 2% | 1% | fused AdamW and gradient clipping over all fp32 parameters |
| Reductions and norms | 1% | 1% | 1% | 1% | |
| Liger loss kernels | <1% | <1% | <1% | <1% | |

At 7.5K, the largest single kernels are:
- a cuBLAS GEMM, 177 ms;
- a generic strided-copy kernel, 177 ms;
- fp32→bf16 casts, 95 ms;
- FA2 backward, 81 ms;
- the depthwise convolution forward, 70 ms;
- a non-vectorized fp32 binary kernel, 68 ms.

At 28K, FA2 backward is the largest kernel (1,097 ms), and FA2 forward adds 702 ms.

### Forward Time by Module (7.5K)

The timers measure only the first forward, not the recompute.

| Module | Count | Forward time |
| --- | ---: | ---: |
| Gated-delta mixers | 24 | 176 ms |
| MLPs | 32 | 111 ms |
| Full-attention mixers | 8 | 68 ms |
| Decoder RMSNorms | 65 | 25 ms |
| Everything else (embedding, Liger loss including its in-forward gradient, other) | | 239 ms |

The MLPs run close to GEMM speed: about 68 TFLOP in 111 ms, roughly 610 TFLOPS. The gated-delta mixers take about 3.5 times their projection GEMM time; the rest goes to the FLA kernels, the convolution fallback and layout copies.

### Weight Precision

The same 7.5K profile with bf16 weights:

| Weights | Step time | Copies and casts | Optimizer | Peak allocated |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 1,834 ms | 490 ms | 86 ms | 67.7 GiB |
| bf16 | 1,897 ms | 652 ms | 55 ms | 35.5 GiB |

fp32 weights cost no step time; the copies are there with either weight type. They therefore come from the model's code path, not from autocasting the weights. The profiler could not attribute them to source lines, because they run on autograd's thread during backward and recomputation, where no Python stack is recorded. The likeliest sources are layout changes in the gated-delta layers:
- the `[B, T, C]` ↔ `[B, C, T]` transposes around the PyTorch convolution;
- `repeat_interleave` of the 16 key heads to the 32 value heads;
- `.contiguous()` calls before the FLA kernels.

### DDP Overhead

With all 4 ranks given nearly equal lengths (7,487 to 7,506 tokens per record), a DDP step took 1,916 ms against 1,842 ms on one GPU: 7,829 against 8,145 tokens/s per GPU, about 4% overhead. The NCCL all-reduce kernels run on their own stream and overlap with backward.

The benchmark's 5,527 tokens/s per GPU is about 30% below the single-GPU profile at the median length. The likely causes, in decreasing order:
1. ranks waiting at each step for the rank with the longest batch;
2. Trainer and data-loader work per step (`logging_steps: 1`, the input-token count gathered every step, collation);
3. a 32-step benchmark epoch amortising start-up worse than a full epoch;
4. the length mix, since short batches run at 6.8K tokens/s.

This gap was not broken down further. A full training epoch with per-rank step times would measure it.

## Opportunities

Ranked by measured share at the median length. None of these is applied in the recommended configuration.

1. **Install `causal-conv1d`** (6% of GPU time, plus some layout copies).
   - It enables Transformers' complete gated-delta fast path. On H100 it added about 5%.
   - It needs a one-time build against CUDA 12.9. Hertz-2 has nvcc 13.3, but a 12.9 compiler can be installed without root, e.g. `nvidia-cuda-nvcc-cu12` from pip.
2. **Find and remove the layout copies** (up to 26%).
   - A trace of one gated-delta layer would show which tensors are copied: with `record_shapes`, or by temporarily disabling checkpointing so the copies run on the Python thread.
   - Fixing them likely means patching the Transformers layer or using FLA's own layer implementation. This is the largest item, and the riskiest.
3. **FlashAttention-3 for the 8 attention layers** (7% at 7.5K, 24% at 28K).
   - `kernels-community/flash-attn3` has a build for Torch 2.11 (`torch211-cxx11-cu128`), built from Dao-AILab's Hopper code.
   - Trying it is a config change (`--attn-implementation`). It mainly helps the long records.
   - `kernels-community/vllm-flash-attn3` is vLLM's inference build and doesn't apply.
4. **Liger SwiGLU and RMSNorm** (part of the 7% elementwise work).
   - First check that Liger's Qwen3.5 patch handles the model's RMSNorm variant and gated MLP exactly.
   - Liger's RoPE likely does not fit this model's interleaved multimodal RoPE.
5. **Freeze the vision tower and MTP head.** This removes their optimizer and clipping work (about 5% at the median length) and lets DDP drop `find_unused_parameters`.
6. **Reduce rank imbalance and per-step Trainer overhead** (the roughly 30% multi-GPU gap). For example: log every 10 steps, drop the per-step input-token count, or batch by token budget so every rank gets a similar amount of work.

Not worth pursuing now:
- **Padding-free packing.** Length grouping already keeps padding at 1.8%. Packing would also need the gated-delta kernels to receive sequence boundaries (`cu_seqlens`) for both the delta rule and the convolution; otherwise state leaks between records.
- **`torch.compile`.** It fits poorly with FLA's Triton kernels and would recompile for every length bucket.

## Environment

Every Qwen3.5 SFT job on Hertz-2 needs these settings.

```bash
# FLA 0.5.2 refuses Triton < 3.7.1 on Hopper (wrong gated-delta gradients, FLA #640); the venv has 3.6.0.
export PYTHONPATH=/mnt/share14T-2/sukhorukov/decomposer_artifacts/kernels/sft/qwen35-overlay/triton${PYTHONPATH:+:$PYTHONPATH}
# Load the pinned FA2 kernel from the local snapshot; per-rank Hub listings hit anonymous 429 rate limits,
# and offline mode rejects the snapshot as incomplete (only the torch211-cxx11-cu128 build is cached).
export LOCAL_KERNELS=kernels-community/flash-attn2=$HOME/.cache/huggingface/hub/kernels--kernels-community--flash-attn2/snapshots/c269cc539ad0c1fc0899abd4b05ecc1303d6c4b1
# Keep FLA's autotuning results across processes.
export TRITON_CACHE_DIR=/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/benchmarks/qwen35-4b-unloop-h200-20261007/triton-cache
export TRITON_CACHE_AUTOTUNING=1
export FLA_TILELANG=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
```

The recommended configuration differs from the FSDP recipe in these training keys:

```yaml
model:
  dtype: float32                    # fp32 weights and Adam state; compute stays bf16 (training.bf16: true)
training:
  per_device_train_batch_size: 2
  global_batch_size: 8
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false
  fsdp: false
  ddp_find_unused_parameters: true  # the vision tower and MTP head get no gradients
```

## Reproducibility

- **Artifacts:** `/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/benchmarks/qwen35-4b-unloop-h200-20261007`, about 1.7 GB:
  - configs, driver, logs, and per-case GPU-process samples (`logs/<case>.gpu`);
  - `runs/<case>/benchmark_summary.json`;
  - the profile outputs, with Chrome traces, under `profile/`;
  - the Triton cache.
- **Benchmark driver:** `run_cases2.sh` (SHA-256 `523af7a236ae9e051656c58c2d7380882ea217ee98f20a11771250383522c61c`). Each case line is `name config per_device_batch stratified|longest count [gate]`. A case is skipped when its gate failed or its summary already exists.
- **Profile script:** `profile_step.py` (SHA-256 `349d49570659f1c3816d0384c8ff2401db7a5cb8a8d5aa246273f5bd9f6275aa`):

  ```bash
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python profile_step.py <model_dir> <release_dir> 7500 <out_dir> [--attribute] [--dtype bfloat16]
  .venv/bin/torchrun --standalone --nproc-per-node=4 profile_step.py <model_dir> <release_dir> 7500 <out_dir> --ddp
  ```

- **Summary table:** `python3 summarize.py <bench_dir>`. It prints exit code, time, throughput, padding, peak memory, loss, the number of processes seen on the GPUs, and the number of autotuning events per case.
- **Configs:**
  - `base.yaml`, the FSDP recipe (SHA-256 `939ef913abd617bfe13336456df9f5a7c24bcdc6892413d5949ee9d9b2a64617`);
  - `ddp_fp32.yaml`, the recommended setup (SHA-256 `c7e5711b8b751ebfbfc81f7570c5a327beca5be42e3fec5300cbbd76dce8e86b`);
  - `noreshard.yaml`, `noac.yaml`, `ddp.yaml` and `ddp_noac.yaml`.

## Problems Encountered

1. **Other users' jobs share Hertz-2.**
   - They start within minutes on any GPU that looks idle, including during our ranks' CPU-side tokenization at start-up. One early one-GPU run shared its GPU with a job at 96% utilisation and measured 275 tokens/s.
   - The driver now samples the GPU processes every 10 s, so a case that shares a GPU is visible and can be discarded. A process seen in only one sample is the previous case's ranks still exiting.
2. **Kernel autotuning.**
   - FLA's L2-norm and gated-norm kernels re-tune for every new 65,536-row bucket, and its cumsum kernels for every batch size. Each tuning takes about 4 seconds, and a new batch size triggers about 30 of them.
   - Without `TRITON_CACHE_AUTOTUNING=1`, every process pays this again. The first FSDP batch-2 run measured 9,364 tokens/s against 19,234 once warm.
3. **Triton version.** The venv's Triton 3.6.0 trips FLA's Hopper correctness guard. The 3.7.1 overlay is required, and it was also used by the September H200 runs.
4. **FA2 kernel loading.** Every rank of every case lists the kernel repository on the Hub, which hit anonymous rate limits (HTTP 429) after a few cases. Offline mode rejects the partial snapshot. `LOCAL_KERNELS` loads the snapshot without network access.
5. **bf16 DDP looked 18% faster at first,** but it trains entirely in bf16 (see **Why bf16 DDP is excluded**). Only DDP with fp32 weights is a fair comparison with FSDP.
6. **Profile attribution.** `with_stack` does not attribute kernels launched on autograd's thread, so the copy kernels show up without a source line.
