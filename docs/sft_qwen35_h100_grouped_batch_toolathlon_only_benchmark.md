# Qwen3.5-4B grouped-batch and Toolathlon-only SFT benchmark

Date: 2026-08-31

## Decision

For the current mixed teacher-prompt dataset on four H100 80 GB GPUs, use the
Hugging Face `kernels-community/flash-attn2` backend, length grouping,
per-device batch 2, and global batch 8. On the actual 192-record mixed
distribution this reduced mean epoch runtime from 100.90 to 68.66 seconds and
raised useful unpadded throughput from 11,902 to 17,491 tokens/s, a 46.96%
increase over the existing HF FA2 batch-1 result. Padding added only 3.80%.

At batch 2, HF FA2 was also decisively better than SDPA: 68.66 versus 128.55
seconds and 17,491 versus 9,342 useful tokens/s. It saved 2.69 GiB of worst-rank
peak allocation and 2.50 GiB of peak reservation. This reverses the batch-1
result, where HF FA2 was slightly slower, and confirms that FA2 becomes useful
after both FLA acceleration and real batching.

Do not use batch 3. The conservative longest-36-record gate failed during the
first backward pass. Rank 0 tried to allocate another 1.68 GiB with only 1.59
GiB free. Although the shortfall shown by the allocator is small, a production
configuration with essentially zero memory margin is not robust to sequence
pairing, allocator state, or library changes, so the full representative
batch-3 run was not attempted.

For the Toolathlon-Gym-only candidate, batch 2 also wins, but narrowly. It
reduced mean runtime from 112.50 to 105.27 seconds and improved useful
throughput from 24,196 to 25,859 tokens/s, or 6.87%. Padding was 2.40%, while
worst-rank peak memory rose from 40.57/48.41 GiB allocated/reserved to
64.25/72.37 GiB. Batch 2 is therefore reasonable on H100 80 GB for this exact
data and runtime, but batch 1 remains the safer fallback if production jobs
show memory variability.

The Toolathlon-only dataset is not an automatic quality improvement. It
concentrates the loss on longer, tool-heavy delegation traces, which may help
Toolathlon specialization, but removes Workplace and GAIA diversity and leaves
only 252 unique training tasks. A future quality comparison should start from
the same checkpoint and prompt, and match supervised-token exposure rather
than epochs. Five mixed epochs contain the same number of target tokens as
approximately 7.86 Toolathlon-only epochs.

No production dependency, environment, training YAML, or committed file was
changed. All implementation and documentation changes remain uncommitted as
requested.

## Dataset candidate

The candidate is an exact, split-preserving view of the Toolathlon-Gym source
already present in the mixed v3 release:

- path:
  `/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/datasets/sft/decomposer-toolathlon-gym-deepseek-qwen35-4b-nonthinking/candidate-v1-493c24c4-teacher-prompt-pass-or-qgt90-32k`;
- candidate fingerprint:
  `8ec9a80182206640567c9b1b3762b331fab62881ccec13889d8249fc0cf02204`;
- parent fingerprint:
  `62d42708828c3dca8cde7dbc2bdfb6fd6a7eecf62050c4b126e00cfb72e62308`;
- selector: environment `toolathlon_gym`, source
  `toolathlon-deepseek-v4-flash-0731-qwen35-4b-nonthinking-n1`;
- 252 train records/groups and 28 validation records/groups, with zero ID or
  group overlap; and
- the parent's teacher prompt, quality policy, prepared token counts, 32K
  limit, message content, and train/validation assignments are unchanged.

The derivation validates the parent manifest fingerprint and prepared-file
checksums before filtering. Selected JSONL lines are copied byte-for-byte in
their original order. The candidate manifest pins its parent identity,
selector, exact ordered train and validation IDs, output checksums, split
groups, and token statistics. It is explicitly marked as a candidate view; a
reviewed final release should be produced only after the surrounding code is
committed.

## Length-distribution change

The source-only data is substantially longer and more target-dense than the
mixed data:

| Train distribution | Records | Mean | p50 | p90 | p95 | Max | Input tokens | Target tokens | Target/input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed | 1,314 | 6,258 | 3,872 | 14,007 | 18,837 | 32,626 | 8,223,580 | 2,320,612 | 28.22% |
| Toolathlon only | 252 | 14,177 | 13,191 | 22,418 | 26,693 | 32,626 | 3,572,641 | 1,475,302 | 41.29% |

Toolathlon contributes only 19.18% of mixed train records, but 43.44% of its
input tokens and 63.57% of its supervised target tokens. At the same five
epochs, source-only training processes 17.86 million input tokens instead of
41.12 million, a 56.56% reduction, and 7.38 million target tokens instead of
11.60 million, a 36.43% reduction.

This changes both optimization and domain coverage:

- positive hypothesis: the model receives a much higher fraction of long,
  multi-tool manager behavior and target tokens per example;
- negative hypothesis: it loses the shorter Workplace examples and GAIA
  execution behavior, and repeatedly sees a much smaller task set; and
- neutral conclusion: throughput and length statistics cannot establish
  downstream quality. That requires an SFT/evaluation ablation, which was
  deliberately not launched in this benchmark.

Matching five mixed epochs by input tokens would require about 11.51
Toolathlon-only epochs, while matching supervised target tokens requires about
7.86. Target-token matching is the more relevant first control because the
loss is computed only on assistant targets, but both token counts should be
reported.

## Benchmark design

The common runtime was Qwen3.5-4B BF16 full-parameter FSDP on four H100 80 GB
GPUs, with activation checkpointing, fused AdamW, Liger fused linear cross
entropy, `fla-core==0.5.2`, `flash-linear-attention==0.5.2`,
`causal-conv1d==1.7.0`, the isolated Triton 3.7.1 overlay, and
`FLA_TILELANG=0`. Only GPUs 0-3 were visible; GPUs 4-7 stayed idle.

The mixed representative sample is the deterministic environment-proportional,
length-stratified 192-record sample used by the preceding benchmarks. It has
1,200,953 useful tokens, lengths from 2,765 to 30,149, p50 3,849, p90 14,168,
and p95 18,874. The newly recorded ordered-record checksum is
`106b2e8dc63a01922f186a527284ed76a6025ce3fa00e736160c43cfe623db6d`.

The Toolathlon sample is a deterministic length-stratified selection of 192
of the 252 training records. It has 2,722,140 useful tokens, lengths from 2,738
to 32,626, p50 13,191, p90 22,418, and p95 26,693. Every repeat and both batch
sizes used ordered-record checksum
`3db3c64c593812cf01ed12cda44a27c392b0ec8cc573eb4151fe57fb5964a5d8`.

For every new configuration, one complete warm-cache run was excluded, then
three complete measured epochs ran sequentially. Throughput below is the fixed
useful-token count divided by the mean trainer runtime. Trainer time excludes
the repeated distributed validation/tokenization startup, but includes all
forward, backward, optimizer, and data-loader work in the epoch.

The existing mixed HF FA2 batch-1 repeats were reused. Two prior grouped SDPA
batch-2 repeats were supplemented with a third new repeat. Three new mixed HF
FA2 batch-2 repeats and three Toolathlon repeats at each batch size completed.

The batch-3 capacity gate selected the longest 36 mixed records: 925,389
useful tokens, lengths 21,751-32,626. At per-device/global batch 3/12 this is
exactly three optimizer steps. It failed on the first backward pass, so neither
a full mixed batch-3 benchmark nor a Toolathlon batch-3 benchmark was run.

## Mixed representative results

| Full-attention backend | Per-device/global batch | R1 / R2 / R3 | Mean +/- stdev | Useful tokens/s | Padding | Worst allocated / reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HF FA2 | 1 / 4 | 99.83 / 98.90 / 103.98 s | 100.90 +/- 2.70 s | 11,902 | 0.00% | 39.64 / 47.57 GiB |
| SDPA | 2 / 8 | 128.14 / 128.94 / 128.59 s | 128.55 +/- 0.40 s | 9,342 | 3.80% | 60.46 / 68.41 GiB |
| HF FA2 | 2 / 8 | 67.30 / 65.93 / 72.75 s | 68.66 +/- 3.60 s | 17,491 | 3.80% | 57.77 / 65.91 GiB |

The actual mixed distribution therefore answers both open questions:

1. batch 2 more than compensates for padding when HF FA2 is used; and
2. after FLA removes the 24-layer linear-attention bottleneck, HF FA2 is highly
   valuable for the remaining eight full-attention layers at batch 2.

The lower batch-2 training loss is not a quality comparison: changing global
batch size changes the number and composition of optimizer steps. This run is
only a systems benchmark.

## Toolathlon-only representative results

| Per-device/global batch | R1 / R2 / R3 | Mean +/- stdev | Useful tokens/s | Padding | Worst allocated / reserved |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 4 | 116.26 / 110.00 / 111.26 s | 112.50 +/- 3.31 s | 24,196 | 0.00% | 40.57 / 48.41 GiB |
| 2 / 8 | 105.80 / 104.22 / 105.79 s | 105.27 +/- 0.91 s | 25,859 | 2.40% | 64.25 / 72.37 GiB |

The source-only data is longer, so batch 1 already drives substantially more
tokens through each step. Batch 2 halves the step count from 48 to 24, but the
24 linear-attention layers, higher activation memory, and a few expensive long
pairs limit the gain to 6.87%. Length grouping keeps padding low enough for the
gain to remain positive.

## Reproducibility

- Repository HEAD before the uncommitted benchmark changes:
  `35dfc10bfb7f9fe87fb1da0efa80b85f60e1d176`.
- Configuration:
  `training/sft/configs/qwen35_4b_nonthinking_mixed_v3_gaia2_execution_110_n7_teacher_prompt_filtered_32k_full_4gpu.yaml`.
- Configuration SHA-256:
  `19502d70f7d7e00ebb284ec96ba429324bfba60d7cdc1435c98790fd9bbf3059`.
- Model revision:
  `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- HF backend: Transformers' supported default version 1 of
  `kernels-community/flash-attn2`, through the Transformers FA2 wrapper.
- Driver SHA-256:
  `75e427d1c7e4884f555b89c37c5ba00f6f48c5a5121872e3598eb77c9b0dc82d`.
- Raw logs, per-run summaries, capacity status, driver, and aggregate:
  `/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/training/sft/benchmarks/qwen35-4b-h100-grouped-batch-toolathlon-only-20260831`.

The representative command shape was:

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
  --per-device-train-batch-size 2 \
  --global-batch-size 8 \
  --attn-implementation kernels-community/flash-attn2 \
  --stratified-train-samples 192 \
  --group-by-length \
  --max-eval-samples 1 \
  --output-dir /absolute/benchmark/output
```

Toolathlon runs additionally supplied the candidate's absolute `--train-file`,
`--validation-file`, and `--manifest-file` paths. The capacity gate used
`--longest-train-samples 36` and batch 3/global 12.

## Problems encountered and resolutions

1. The production loader previously took dataset paths only from YAML. Added
   explicit train, validation, and manifest CLI overrides so a candidate can
   be benchmarked without modifying the production configuration.
2. Earlier summaries recorded the sampling strategy and token totals but not
   exact sample identity. New summaries record ordered IDs, their checksum,
   and a checksum over IDs plus prepared input/target lengths. This caught
   accidental sample drift across repeats and batch sizes.
3. `CanonicalSource.partition` describes source provenance and remains
   `train` even when the prepared builder assigns a record to validation. The
   candidate derivation therefore preserves effective split membership from
   the parent files and validates disjoint group IDs, instead of incorrectly
   rewriting provenance.
4. The first candidate draft omitted the `records` field inside inherited
   token-stat subobjects. It was caught during manifest inspection, moved to a
   recoverable sibling named `candidate-...-draft-missing-stat-counts`, and the
   intended candidate path was regenerated with complete statistics.
5. The first resumable benchmark driver stopped after the expected batch-3
   failure before starting source-only runs. The gate was made explicitly
   non-fatal: it writes an exit-code sidecar, skips all batch-3 measurements,
   and continues independent batch-1/batch-2 cases. Completed summaries are
   skipped on resume.
6. Batch 3 missed the allocation by only about 90 MiB, but that is not usable
   capacity margin. The failure was retained as the result rather than hiding
   it with allocator tuning or a less conservative sample.
7. Every torchrun rank still validates and tokenizes the complete prepared
   split with `num_proc=8` before TRL tokenizes the selected sample again. Four
   ranks can launch 32 workers per map. This cost is outside trainer runtime
   and is conspicuous on a 252-record source-only dataset. Shared cached
   validation/tokenization remains the next end-to-end startup optimization.
8. Transformers warned that a newer Hub kernel version exists but selected its
   supported default version 1. The default was retained so this result stays
   comparable with the preceding HF FA2 benchmark.
9. `torchrun` emitted harmless IPv6 address-family warnings. All successful
   runs initialized four ranks, wrote summaries, exited normally, and released
   GPU memory.

## Uncommitted support

`data/sft/derive_source_view.py` adds the checksum-pinned, atomic candidate
derivation described above. `training/sft/train.py` adds candidate path
overrides and exact benchmark sample identity. Tests cover exact source
selection, split preservation, fingerprint validation, immutability, CLI path
resolution, and deterministic ordered sample hashes.

These helpers are intentionally not committed yet. The candidate is suitable
for review and benchmark reuse, but should not be promoted to a final immutable
training release until the code and manifest have passed review in a clean
commit.

## Verification

- `CUDA_VISIBLE_DEVICES='' uv run --with-requirements
  training/sft/requirements-mlspace.txt pytest -q tests/test_*sft*.py`: 162
  passed, with two existing multiprocessing/fork deprecation warnings.
- Candidate validation recomputed the manifest fingerprint and prepared-file
  checksums, parsed all 280 canonical records, matched every ordered ID, and
  found zero train/validation ID or group overlap.
- All 12 new successful measured/warm runs plus the new SDPA control wrote 13
  benchmark summaries. The batch-3 gate retained its expected nonzero status
  and OOM log instead of producing a false success summary.
- Ruff formatting and standard `E`, `F`, and import checks passed; the full
  repository rule set additionally reports existing exception-type and
  long-string preferences outside this benchmark's scope.
- Python byte compilation, aggregate JSON parsing, and `git diff --check`
  passed.
- Final `nvidia-smi`: all eight GPUs at 0 MiB and 0% utilization, with no
  `torchrun` or `training.sft.train` process remaining.
