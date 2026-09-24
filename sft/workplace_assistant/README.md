# Workplace Assistant SFT releases

The benchmark-neutral canonical schema, builder, and NeMo-Gym adapter are
shared in `sft/`. The Workplace-specific build specifications (`specs/`) and
their public command live here:

```bash
uv run --group train python -m sft.workplace_assistant.prepare \
  --dataset workplace-all-v3

uv run --group train python -m sft.workplace_assistant.prepare \
  --dataset workplace-26b-nonthinking-v3

uv run --group train python -m sft.workplace_assistant.prepare \
  --dataset workplace-deepseek-e4b-thinking-v1
```

Build the new non-thinking Gemma-4 E4B releases from the same normalized trace
pool with explicit prepared-token ceilings:

```bash
uv run --group train python -m sft.workplace_assistant.prepare \
  --dataset workplace-deepseek-e4b-thinking-v2-8k

uv run --group train python -m sft.workplace_assistant.prepare \
  --dataset workplace-deepseek-e4b-thinking-v2-32k
```

Both releases apply reward and trace-validity filtering, assign prompt groups to
train or validation once, remove teacher reasoning for tokenization, render the
prompt profile selected by the dataset specification with the Gemma-4 E4B
training template, and then apply the inclusive token ceiling. Consequently,
`v2-8k` is a strict subset of
`v2-32k`, and every shared record keeps the same ID and split. Raw rollout
artifacts are never removed.

Each retained row carries its prepared token and supervised-token counts. The
manifest pins the tokenizer revision and chat-template hashes and records raw
and retained length statistics plus every overlength exclusion. Releases are
immutable; preparation refuses to replace an existing version directory.

Releases are written under `/home/sukhorukov/decomposer_artifacts/datasets/sft/<dataset-id>/<version>/`.
