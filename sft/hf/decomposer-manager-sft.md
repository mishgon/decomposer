---
pretty_name: "Decomposer Manager SFT (tau2, Workplace, Toolathlon, WideSeek)"
tags:
- agents
- tool-use
- multi-agent
- distillation
- sft
configs:
- config_name: "1.0.0"
  default: true
  data_files:
  - split: train
    path: 1.0.0/train.jsonl
  - split: validation
    path: 1.0.0/validation.jsonl
---

# Decomposer Manager SFT

Supervised fine-tuning data for the manager of Decomposer, a multi-agent harness. The manager orchestrates subagents through four tools (`new`, `fork`, `run`, `wait`) and never calls environment tools itself. Each record is one complete manager conversation that a teacher model produced on a task from one of four gyms. Releases are tokenizer-free (messages plus tools), so any chat model with tool calling can train on them; they were built for a Qwen3.5-4B non-thinking student.

## Versions

| Version | Folder and tag | Records (train / validation) | Gyms | Built | Internal build |
|---|---|---|---|---|---|
| **1.0.0** | `1.0.0/`, tag `v1.0.0` | 4,220 (3,797 / 423) | tau2, Workplace, Toolathlon, WideSeek | 2026-10-08 | `v3-tau2-workplace-toolathlon-wideseek-20261008` |

Each version folder holds `train.jsonl`, `validation.jsonl` and `manifest.json`. The manifest records the full provenance: source snapshot digests, selection and split rules, per-source counts and file hashes. Pin a version with its tag, for example `revision="v1.0.0"`.

**1.0.0:**
- **Fingerprint:** `50a733439db998f83c559946eb5f57fc77c811916142b787bdf2245d1fdb8df4`.
- **Build:** spec `sft/specs/decomposer_mixed_qwen38_qwen35_4b_unloop_nonthinking_v3.yaml` at commit `6716e23`; adapters `nemo_gym` 8, `toolathlon_gym` 10, `wideseek` 1.
- **Manifest label:** the manifest still names the internal build `v3-…`, because renaming it would change the fingerprint.

## Versioning

Versions follow MAJOR.MINOR.PATCH. The bump is decided when each release is cut.

- **MAJOR:** a change that breaks training or comparisons: a new record format, a different system prompt, tools or core interface, a different teacher or harness setup, or a reassigned train/validation split.
- **MINOR:** more data that stays compatible: new snapshots, finished collections, a new gym, or optional new fields. Existing records keep their split.
- **PATCH:** fixes without new data: dropping or correcting a few records, metadata or card fixes. The split does not change.

A published version folder never changes; every change becomes a new version with its own folder and tag. This card is the only file that changes between releases.

## Changelog

- **1.0.0** (published 2026-10-09): first published release. It is internal build `v3-tau2-workplace-toolathlon-wideseek-20261008`, the data the October 2026 LoRA runs were trained on. The Toolathlon and WideSeek collections in it are partial; see Partial Data and Known Issues.

## Models

- **Teacher (manager):** Qwen3.8-Flash-Next, non-thinking. tau2 used an FP8 build; Workplace used NVFP4; Toolathlon and WideSeek used NVFP4, then FP8 from 2026-10-08.
- **Subagents:** Qwen3.5-4B-unlooped in thinking mode. Their internal tool calls and reasoning are not in the records: the manager only sees each run's final reply.
- **System prompt:** the Decomposer teacher prompt, identical in every record (sha256 `26a6ee71…`).
- **No reasoning:** assistant targets are plain text and tool calls only.

## How It Was Collected

| Gym | Task source | Rollouts | Kept | Selection |
|---|---|---|---|---|
| tau2 | `decomposer_broad_v1` pool: 4,843 runnable `tasks_hard` tasks outside the held-out sets, without ask tasks or DB-order-sensitive ones | 1 per task, 2026-10-07/08 | 2,948 | reward exactly 1 |
| Workplace | NeMo Gym Workplace Assistant, train split (1,255 tasks) | 1 per task, 2026-10-06 | 988 | reward 1 (the final environment state equals the expected state) |
| Toolathlon | Toolathlon coverage collection | 1,095 episodes finished by 2026-10-08 19:41 UTC | 214 (one per task) | no agent error, and a binary pass or a check fraction above 0.9 |
| WideSeek | WideSeek coverage collection | 253 attempts finished by 2026-10-08 19:44 UTC | 70 (one per task) | score 1, or above 0.9 |

What the manager receives in each gym:

- **tau2:** only the user's request. The domain policy goes to the subagents' system prompt; the manager never sees it.
- **Workplace:** a date line ("Today's date is …") followed by the request.
- **Toolathlon and WideSeek:** the task text.

## Partial Data

Two kinds of data in this release are partial.

1. **Partial collections.** The Toolathlon and WideSeek collections were still running when they were snapshotted. They contain only what had finished by 2026-10-08 around 19:40 UTC, and tasks without a qualifying episode by then are missing.
2. **Partial successes.** 164 records have a reward below 1, though above 0.9:
   - Toolathlon: 110 of 214 records, with check fractions from 0.902; the other 104 are full passes.
   - WideSeek: 54 of 70 records; the other 16 score 1.

   `outcome.reward` holds the score, and `outcome.metrics` holds the details (`binary_pass` and `check_fraction` where available). Filter on `outcome.reward == 1` for strict successes.

tau2 and Workplace records are complete successes (reward 1) of complete collections.

## Record Format

JSONL, one record per line:

- **`id`, `group_id`, `schema_version`:** `group_id` is the task group used for the split; `schema_version` is 1.
- **`messages`:** system, user, then assistant and tool turns, ending with the final assistant answer.
  - **Assistant:** `{role, content, tool_calls: [{id, type: "function", function: {name, arguments}}], teacher_reasoning: null}`. `arguments` is a JSON object. Several calls in one message are kept as the teacher emitted them.
  - **Tool:** `{role, content, tool_call_id, name}`, in the order the harness returned them.
- **`tools`:** the four manager tools in OpenAI chat format. The `new` tool embeds the gym's table of subagent types.
- **`source`:** adapter, benchmark, environment, source ID, task ID, rollout ID and teacher.
- **`outcome`:** success, reward and metrics.
- **`attributes`:** category (a tau2 domain, `workplace_assistant_<app>`, `toolathlon_gym` or `wideseek`) plus collection details.

**Teacher mistakes are kept on purpose.** When the harness answered a mistake, the record keeps the call or answer together with the harness's reply. That covers:
- refused calls (a busy agent, an unknown agent or agent type, several runs of one agent in one turn, fork and run of one agent, `wait` alongside other calls);
- calls of tools that don't exist;
- premature answers, which are followed by an injected user message ("You cannot respond to the user until you have received results from all started runs…").

They make up about 4% of assistant tokens. Mask them if you do not want them as training targets.

## Splits

`prompt_fixed`: 10% of task groups go to validation, with per-category quotas and seed 42.
- **Groups:** tau2 and Workplace are grouped by the gym input, so a tau2 domain and its `_dsh` variant with the same prompt share a group. Toolathlon and WideSeek are grouped by task.
- **No overlap:** no task, group or prompt appears in both splits.

| Gym | Train | Validation |
|---|---|---|
| tau2 (202 domains) | 2,652 | 296 |
| Workplace | 889 | 99 |
| Toolathlon | 193 | 21 |
| WideSeek | 63 | 7 |

## Size

Measured with the Qwen3.5 tokenizer and chat template:

- **Total:** 38.4M tokens, 8.86M of them in assistant turns.
- **Median per record:** tau2 6.8K, Workplace 6.0K, Toolathlon 29.3K, WideSeek 20.5K. The maximum is 65.5K.
- **Toolathlon share:** 5% of the records but 27% of the assistant tokens.
- **Truncation:** at a 16K training length, 89% of Toolathlon records and 64% of WideSeek records are cut, so they lose their final answers.

## Loading

Read the JSONL directly. Automatic schema inference in `datasets` turns the nested `messages` and `tools` into Arrow structs that add null keys, which changes the chat-template output.

```python
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "decomposer-datasets/decomposer-manager-sft",
    "1.0.0/train.jsonl",
    repo_type="dataset",
    revision="v1.0.0",
)
records = [json.loads(line) for line in open(path, encoding="utf-8")]
```

To render a record, drop the null `teacher_reasoning` key and call `tokenizer.apply_chat_template(record["messages"], tools=record["tools"], enable_thinking=False)`.

## Known Issues (Audit of 2026-10-09)

- **Repetition loop.** `nemo_gym:tau2_gym:tau2-qwen38-flash-fp8-nonthinking-broad-v1-n1:652:0` has an 82K-character repetition loop as its final answer. Drop it.
- **Nudge records.** 16 records contain the injected nudge, after which Qwen's template renders the earlier assistant turns without the empty think block.
- **Self-reviews.** About one record in five has a review run by the agent that did the work, or by a fork of it, despite the system prompt's review rule.
- **Answer format.** About 20% of tau2 final answers add text to a requested terse format (a single word, only an ID). They still scored reward 1.
- **Parallel calls in Workplace.** Workplace requests set `parallel_tool_calls: false`. The teacher's endpoint ignored it, but vLLM enforces it, so the 72 Workplace turns with parallel calls cannot be reproduced under vLLM unless the flag is changed.
- **Sandbox identifiers.** Toolathlon and WideSeek records contain URLs and object IDs from the collection sandboxes.
- **Leaked key.** The source traces of the Toolathlon collection contain a leaked proxy key inside subagent runs. This release contains no key value: it was checked against every value found in the source. One record mentions the variable name only. Do not publish the source snapshots.
