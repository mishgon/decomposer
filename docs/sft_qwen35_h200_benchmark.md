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

**Update, same day.** **Optimization Benchmark** applied five of those opportunities to full training and to LoRA. The steady-state rate (excluding each run's first step) of full training rose from 27,076 to 35,128 tokens/s (+30%). LoRA rose from 27,442 to 36,894 tokens/s (+34%), or to 39,475 with adapter dropout 0. Use its **Recommended Configurations**.

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

This gap was not broken down further. A full training epoch with per-rank step times would measure it. (**Optimization Benchmark** later measured per-step times. Excluding the first step, the gap is 17%, and the rank imbalance is 1.083.)

## Opportunities

Ranked by measured share at the median length. **Optimization Benchmark** measures each of them.

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
5. **Freeze the vision tower and MTP head.** This lets DDP drop `find_unused_parameters`. It was also expected to remove their optimizer and clipping work (about 5% at the median length), but the optimizer already skips them (see **Optimization Benchmark**).
6. **Reduce rank imbalance and per-step Trainer overhead** (the roughly 30% multi-GPU gap). For example: log every 10 steps, drop the per-step input-token count, or batch by token budget so every rank gets a similar amount of work.

Not worth pursuing now:
- **Padding-free packing.** Length grouping already keeps padding at 1.8%. Packing would also need the gated-delta kernels to receive sequence boundaries (`cu_seqlens`) for both the delta rule and the convolution; otherwise state leaks between records.
- **`torch.compile`.** It fits poorly with FLA's Triton kernels and would recompile for every length bucket.

## Optimization Benchmark

Later the same day, all six opportunities were measured in both training modes, and five went into the recommended configurations:
- **Full training:** the recommended DDP setup above (fp32 weights, bf16 autocast).
- **LoRA:** a bf16 base with fp32 adapters (r 32, alpha 64, dropout 0.05) on the 200 linear projections of the language model, also on DDP (`sft/README.md`, **LoRA**).

Together the changes speed up full training by 30% and LoRA by 34%; with LoRA adapter dropout set to 0, LoRA gains 44%. Every change keeps the loss trajectory. Of the sixth opportunity, the Trainer's logging settings made no difference. Balancing ranks needs variable-size batches, which this benchmark did not try.

### Method

- **Cases:** the same 256-record sample, on 4 GPUs with batch 2 per GPU (global batch 8), with Decomposer at `2a1a609` (`feat/sft_lora`). Each case adds one change to its mode's baseline; the stack adds all of them.
- **Metric:** steady-state useful tokens/s from the trainer's step timing (benchmark mode). It counts the real tokens of steps 4 to 32 and divides by their wall time. The first step costs about 10 s of start-up in every case, so the end-to-end rate in **Benchmark Results** (22,108 tokens/s for this baseline) is lower than the steady rate (27,076).
- **Noise:** each baseline ran twice. The full baseline measured 26,784 and 27,369 tokens/s (2.2% apart), LoRA 27,274 and 27,609 (1.2% apart). Gains below are against the mean of the two; a single-run difference under about 3% is within noise.
- **Warm cache:** an untimed run of each mode's stack first filled the Triton autotuning cache. No timed case autotuned, and no timed case shared a GPU with another job.

### Correctness Gates

Before any timed run, a gate compared each kernel with the code it replaces, forward and backward in bf16. The largest relative error over the output and all gradients:

| Change | Gate | Worst relative error | Limit |
| --- | --- | ---: | ---: |
| FA3 | Tiny Qwen3.5 (head dim 256, 4 query and 1 KV head, padded batch), loss and all gradients against FA2 | 9.8e-3 | 2e-2 |
| `causal-conv1d` | Strided `[B, C, T]` input as Transformers passes it, against PyTorch's depthwise convolution with SiLU | 4.1e-3 | 1e-2 |
| Liger RMSNorm | Non-zero weights, so the `(1 + weight)` offset matters, against `Qwen3_5RMSNorm` | 5.3e-5 | 1e-2 |
| Liger SwiGLU | Width 9,216, against `Qwen3_5MLP` | 3.5e-3 | 1e-2 |
| Native GVA | `tests/test_sft_train_options.py`: one gated-delta layer, outputs and input gradients against repeated key heads | passes | 2e-2 |

The gates use bf16 weights. In full training the convolution weight stays fp32 while its input is bf16. `causal-conv1d` dispatches on both types, and the run-level checks below cover that case.

Every timed case matched its baseline's first step (loss within 0.0008, gradient norm within 0.2). Over 10-step windows, the stacks' mean losses differ from the baselines' by at most 0.0034. That is less than the two full stack runs differ from each other (0.0046 at steps 11-20), and bf16 kernels are not bitwise deterministic:

| Steps | 2-10 | 11-20 | 21-30 |
| --- | ---: | ---: | ---: |
| Full baselines | 0.5774-0.5780 | 0.4745-0.4754 | 0.4671-0.4682 |
| Full stack | 0.5783 | 0.4716-0.4762 | 0.4669-0.4689 |
| LoRA baselines | 0.5949-0.5954 | 0.4877-0.4881 | 0.4950-0.4955 |
| LoRA stack | 0.5948-0.5949 | 0.4884-0.4893 | 0.4952-0.4960 |
| LoRA stack, dropout 0 | 0.5945 | 0.4877 | 0.4946 |

### Results

| Change | Full tokens/s | Full gain | LoRA tokens/s | LoRA gain |
| --- | ---: | ---: | ---: | ---: |
| Baseline (mean of 2 runs) | 27,076 | | 27,442 | |
| `causal-conv1d` | 31,565 | +16.6% | 31,860 | +16.1% |
| Native GVA (`model.qwen35_native_gva`) | 27,715 | +2.4% | 28,064 | +2.3% |
| FA3 | 25,460 and 29,031 | -6.0% and +7.2% | 28,567 | +4.1% |
| Liger RMSNorm and SwiGLU | 27,958 | +3.3% | 29,163 | +6.3% |
| Freeze the vision tower, `ddp_find_unused_parameters: false` | 27,605 | +2.0% | (LoRA freezes the base) | |
| Log every 10 steps, no input-token count | 27,655 | +2.1% | 27,476 | +0.1% |
| LoRA adapter dropout 0 instead of 0.05 | | | 28,611 | +4.3% |
| **All of the above except dropout**, 2 runs | **35,052 and 35,204** | **+29.7%** | **36,246 and 37,542** | **+34.4%** |
| All, but FA2 instead of FA3 | 33,493 | +23.7% | | |
| All, with LoRA dropout 0 | | | 39,475 | +43.9% |

Peak allocated memory fell with the stack from 91.7 to 90.0 GiB in full training, and from 34.4 to 31.8 GiB in LoRA (28.6 GiB with dropout 0).

**`causal-conv1d` is the largest single gain** in both modes. It replaces PyTorch's depthwise convolution and reads the transposed input in place.

**The small changes add up.** In full training, native GVA, Liger, FA3 and freezing each sit near the noise level on their own. Yet the full stack gains 30% where `causal-conv1d` alone gains 17%. Inside the stack, FA3 adds 4.9% over FA2 (35,128 against 33,493 tokens/s).

**FA3 in full training.** The first FA3 run lost one step: step 9, the 40K-token step, took 13.7 s against 5.4 s. The repeat did not. In both runs every other long step ran faster than in either baseline run (for example 4.0 s against 4.4 and 5.4 s at step 25). FA3 halves attention time in the profile below.

**LoRA runs barely faster than full training,** at 27,442 against 27,076 tokens/s. LoRA skips the weight-gradient GEMMs and the optimizer, but PEFT adds casts and elementwise work of about the same size; **Step Profile After the Changes** has the breakdown.

**LoRA adapter dropout.** Lowering it from 0.05 to 0 gains 4.3% at baseline and 7.0% in the stack, and saves 3.2 GiB. Dropout is a regularisation choice, not only a speed setting. Over these 32 steps, the loss did not change without it, but that says nothing about generalisation.

**Logging** every step costs nothing measurable: in full training, the 32 step times sum to 88.8 s with `logging_steps: 10` against 88.1 s with `logging_steps: 1`.

**Freezing the vision tower saves no optimizer time.** Its parameters never receive gradients, so fused AdamW and gradient clipping already skipped them. The profile shows 86 ms of optimizer kernels per step with or without freezing, so the 5% estimate in **Opportunities** was wrong. Freezing still lets DDP skip its search for unused parameters and saves 1.2 GiB, at no cost.

**Rank imbalance** is 1.083 in every case, because every case sees the same batch order. Hugging Face's length-grouped sampler splits the 256 records into 4 blocks of 64 and sorts each block longest first. Steps 1, 9, 17 and 25 therefore carry each block's longest records. Apart from step 1, which also carries start-up, those steps take 3 to 6 s against 1.3 to 2 s for the rest. Within each step, rank 0 always gets the two longest records.

With two padded records per rank, pairing records of adjacent length already minimises the slowest rank's padded tokens. A sampler that balances token sums at a fixed batch size therefore cannot shorten a step. Balancing would need a different number of records per rank (token-budget batches). On this sample, perfect balancing would save at most about 8%.

### Step Profile After the Changes

`profile_step.py` now takes each change as an option (`--attn`, `--gva`, `--liger-all`, `--freeze-visual`) plus a LoRA mode (`--lora`, `--lora-dropout`, `--no-lora-input-cast`). As before, it runs one GPU, batch 2 of real records, 3 warm-up and 5 timed steps.

| Profile | Records near | Step time | Tokens/s (1 GPU) | Gain | Peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full baseline | 7.5K | 1,842 ms | 8,145 | | 67.7 GiB |
| Full stack | 7.5K | 1,382 ms | 10,859 | +33% | 66.9 GiB |
| LoRA baseline (dropout 0.05) | 7.5K | 1,893 ms | 7,923 | | 16.9 GiB |
| LoRA stack, dropout 0.05 | 7.5K | 1,369 ms | 10,960 | +38% | 16.3 GiB |
| LoRA stack, dropout 0 | 7.5K | 1,326 ms | 11,312 | +43% | 15.4 GiB |
| LoRA stack, dropout 0, no PEFT input cast | 7.5K | 1,123 ms | 13,360 | +69% | 14.7 GiB |
| Full baseline | 28K | 7,483 ms | 7,438 | | 82.1 GiB |
| Full stack | 28K | 5,189 ms | 10,725 | +44% | 79.5 GiB |
| LoRA baseline (dropout 0.05) | 28K | 8,048 ms | 6,916 | | 38.2 GiB |
| LoRA stack, dropout 0 | 28K | 5,310 ms | 10,481 | +52% | 32.4 GiB |
| LoRA stack, dropout 0, no PEFT input cast | 28K | 4,705 ms | 11,828 | +71% | 29.8 GiB |

GPU kernel time per step at 7.5K, in ms:

| Category | Full baseline | Full stack | LoRA baseline | LoRA stack (dropout 0) | LoRA stack, no input cast |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEMM | 736 | 744 | 579 | 580 | 584 |
| Copies and casts | 490 | 242 | 563 | 257 | 98 |
| Elementwise | 138 | 53 | 260 | 156 | 157 |
| FlashAttention | 137 | 70 | 136 | 68 | 70 |
| FLA gated delta | 131 | 120 | 129 | 115 | 118 |
| Short convolution | 120 | 17 | 98 | 17 | 17 |
| Optimizer | 86 | 86 | 2 | 2 | 2 |
| Reductions and norms (with Liger RMSNorm) | 28 | 40 | 21 | 31 | 31 |
| Other (Liger SwiGLU, dropout) | 1 | 26 | 68 | 26 | 26 |
| **Total** | **1,869** | **1,401** | **1,859** | **1,253** | **1,104** |

**Copies.** The changes halve copy time. The 177 ms generic strided-copy kernel, the largest copy at baseline, disappears. Its likely main sources were the transposes around the PyTorch convolution and the `repeat_interleave` of key heads: `causal-conv1d` reads the strided input in place, and native GVA skips the repeat. No profile isolates either change.

**Attention.** FA3 halves attention time at both profiled lengths: from 137 to 70 ms at 7.5K, and from 1,806 to 920 ms at 28K.

**GEMMs** are now 53% of the full stack's kernel time at 7.5K, against 39% before; the step is mostly GEMM-bound.

**The PEFT input cast.** PEFT casts each adapted module's bf16 input to the fp32 adapter dtype (`_cast_input_dtype`). Under bf16 autocast, the adapter matmul then runs in bf16 anyway, so the cast is wasted work in 200 modules, during forward, recomputation and backward. Turning it off (`cast_input_dtype_enabled = False` on each LoRA layer, as `peft.helpers.disable_input_dtype_casting` does) cuts copies from 257 to 98 ms: +18% at 7.5K and +13% at 28K. With dropout 0 the computation is unchanged. The trainer does not do this yet.

**The multi-GPU gap** is smaller than it first looked. On the steady metric, the full baseline runs 6,769 tokens/s per GPU, 17% below the single-GPU 7.5K profile (8,145); the full stack runs 8,782, 19% below its profile (10,859). The 30% gap in **DDP Overhead** used end-to-end time, which includes the first step's start-up. The remaining gap comes from the length mix, rank imbalance and about 4% of DDP overhead.

### Recommended Configurations

Both modes need the environment in **Environment**. That includes `causal-conv1d` on `PYTHONPATH`: without it, Transformers falls back to the PyTorch convolution and only logs a warning. Benchmark runs record the convolution actually used in `benchmark_summary.json` under `linear_attention_runtime.causal_conv1d`.

Full training, changes against the DDP baseline in **Environment**:

```yaml
model:
  dtype: float32
  attn_implementation: kernels-community/flash-attn3@3c1451f803c146c54222305a8c350f7f46bb5135
  qwen35_native_gva: true
  freeze_modules: [model.visual]
training:
  liger_kernel_config:
    fused_linear_cross_entropy: true
    cross_entropy: false
    rms_norm: true
    swiglu: true
    rope: false
  ddp_find_unused_parameters: false
```

LoRA uses the same keys, except:

```yaml
model:
  dtype: bfloat16      # the frozen base; PEFT keeps the adapters in fp32
  # no freeze_modules: LoRA already freezes the whole base
lora:
  r: 32
  alpha: 64
  dropout: 0.0         # 7% faster than 0.05 in the stack; choose for regularisation, not speed
```

Logging settings make no measurable difference; keep `logging_steps: 1` for per-step loss curves.

### Next Opportunities

1. **Skip PEFT's input cast in LoRA under bf16 autocast** (+13% to +18% per step in the profile). It is a small trainer change, but it still needs a benchmark across all 4 GPUs.
2. **Token-budget batches** to balance ranks (at most about 8% on this sample).
3. **The remaining copies in full training** (242 ms at 7.5K, 17% of kernel time). As before, the profiler cannot attribute them to source lines.

## Environment

Every Qwen3.5 SFT job on Hertz-2 needs these settings.

```bash
# Triton 3.7.1 and the sm_90 causal-conv1d build. FLA 0.5.2 refuses Triton < 3.7.1 on Hopper
# (wrong gated-delta gradients, FLA #640); the venv has 3.6.0.
BUNDLE=/mnt/share14T-2/sukhorukov/decomposer_artifacts/kernels/sft/qwen35-hf-fa2-fla-h200-v1
export PYTHONPATH=$BUNDLE/triton:$BUNDLE/causal${PYTHONPATH:+:$PYTHONPATH}
# Load the pinned FA2 and FA3 kernels from local snapshots; per-rank Hub listings hit anonymous 429 rate limits,
# and offline mode rejects the snapshots as incomplete (only the torch211-cxx11-cu128 builds are cached).
SNAP=$HOME/.cache/huggingface/hub
export LOCAL_KERNELS=kernels-community/flash-attn2=$SNAP/kernels--kernels-community--flash-attn2/snapshots/c269cc539ad0c1fc0899abd4b05ecc1303d6c4b1:kernels-community/flash-attn3=$SNAP/kernels--kernels-community--flash-attn3/snapshots/3c1451f803c146c54222305a8c350f7f46bb5135
# Keep FLA's autotuning results across processes.
export TRITON_CACHE_DIR=/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/benchmarks/qwen35-4b-unloop-h200-20261007/triton-cache
export TRITON_CACHE_AUTOTUNING=1
export FLA_TILELANG=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
```

The DDP baseline differs from the FSDP recipe in these training keys; **Recommended Configurations** lists the changes on top of it:

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

- **Artifacts:** `/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/benchmarks/qwen35-4b-unloop-h200-20261007`, about 12 GB:
  - configs, drivers, logs, and per-case GPU-process samples (`logs/<case>.gpu`);
  - `runs/<case>/benchmark_summary.json`, plus the LoRA smoke run's exported adapter and merged model (8.8 GB, `runs/lora_smoke`);
  - the profile outputs, with Chrome traces, under `profile/`;
  - the Triton cache.
- **Kernels:** the runtime bundle `/mnt/share14T-2/sukhorukov/decomposer_artifacts/kernels/sft/qwen35-hf-fa2-fla-h200-v1` holds Triton 3.7.1 and `causal-conv1d` 1.7.0 built for sm_90. `python -m sft.prepare_qwen35_fast_runtime --cuda-home /mnt/share14T-2/sukhorukov/decomposer_artifacts/toolchains/cuda-12.9.86 --bundle-dir <bundle>` built it, and `--verify-only` checks it. FA3 is `kernels-community/flash-attn3` at revision `3c1451f803c146c54222305a8c350f7f46bb5135`, downloaded once into the Hugging Face cache.
- **Benchmark driver:** `run_cases2.sh` (SHA-256 `523af7a236ae9e051656c58c2d7380882ea217ee98f20a11771250383522c61c`). Each case line is `name config per_device_batch stratified|longest count [gate]`. A case is skipped when its gate failed or its summary already exists.
- **Profile script:** `profile_step.py` (SHA-256 `8b58216e583b498c3d2cceefbab0220b90aa3c2a4d98a94214efe069efab1909`). Its defaults reproduce the first profiles, which ran an earlier version (`349d4957…`) that named the convolution category `conv1d_fallback`:

  ```bash
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python profile_step.py <model_dir> <release_dir> 7500 <out_dir> [--attribute] [--dtype bfloat16]
  .venv/bin/torchrun --standalone --nproc-per-node=4 profile_step.py <model_dir> <release_dir> 7500 <out_dir> --ddp
  # Optimization benchmark: the full and LoRA stacks (causal-conv1d comes from PYTHONPATH).
  ... profile_step.py <model_dir> <release_dir> 7500 <out_dir> --attn <FA3> --gva --liger-all --freeze-visual
  ... profile_step.py <model_dir> <release_dir> 7500 <out_dir> --lora --lora-dropout 0.0 [--no-lora-input-cast] --attn <FA3> --gva --liger-all
  ```

- **Optimization benchmark:**
  - `gates.py` (SHA-256 `3728caf25a1d16121c86652321dd5090cb1b72a17efb464a8ca8d8e1d718fb3a`) runs the correctness gates on one GPU.
  - `run_cases3.sh` (SHA-256 `93057ebedf7b94968efbf7adb214ac84242bd635638b4349315b13d965777ace`) runs the cases. Its case lines are `name config per_device_batch stratified|longest count plain|causal`, where `causal` adds the bundle's `causal-conv1d` to `PYTHONPATH`. It ran `round2_warm.txt`, `round2.txt` and `round3.txt`.
  - `summarize2.py <bench_dir> <case>...` prints steady throughput, step-time median, rank imbalance, peak memory, first-step loss and gradient norm, the convolution and attention actually used, GPU processes and autotuning events.
  - `run_profiles4.sh` and `run_profile_nocast.sh` ran the profiles, and `compare_profiles.py <profile_dir> <name>...` tabulates them.
  - Configs: `ddp_fp32.yaml` and `lora.yaml` (SHA-256 `f0079bd7c880aad057e12cb96b4a8ca05a92b9e89da9cc35889f7fe48af9c455`) are the baselines. `f_<change>.yaml` and `l_<change>.yaml` add one change each. The stacks are `f_all.yaml` (`113caa5827b0c1a495d510d88353bf6b35342ca36132aad7c9018f78f6ed5f4d`), `l_all.yaml` (`803ff5a39921aac8ad4bab107e8e3d95002ff52e051a298f09641508bb210ec1`) and `l_all_nodrop.yaml` (`2d9a5002742d70c5ed6f80411e5f21abbc1330c79aac135cecc37e9629cf2e78`).

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
7. **Building `causal-conv1d`.** It needs a CUDA 12.9 `nvcc` to match Torch's CUDA. Hertz-2 has only 13.3, and the pip wheel `nvidia-cuda-nvcc-cu12` 12.9 ships `ptxas` without `nvcc`. A user-space micromamba install of conda-forge's `cuda-nvcc=12.9.86` and `cuda-cudart-dev=12.9` provided the toolchain (`toolchains/cuda-12.9.86`).
8. **The FA3 revision.** The pin must be the kernel repository's revision from `https://huggingface.co/api/kernels/kernels-community/flash-attn3`. The model API's `sha` for the same repository is not a valid kernel revision.
9. **Shared configuration objects.** The first FA3 gate built both models from one config object. Transformers reads the attention backend from the config when attention runs, so both models ran FA3 and the gate compared FA3 with itself. Each model now gets its own copy of the config.
