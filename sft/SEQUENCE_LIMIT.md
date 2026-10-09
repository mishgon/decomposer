# Shared Sequence Filtering

`StudentSequenceFilter` in `sft/filtering.py` is the single length check for
individual raw traces, batch filtering, and live collection. It renders the
complete teacher conversation with the student tokenizer, system message, and
tool schemas. Reasoning is excluded; raw messages remain unchanged. The rendered
sequence includes tool observations and assistant tool calls. It has no generation
prompt. Keep sequences of at most **32,768** tokens.

`StudentSequenceLimit` adds this check after each teacher response. The response
is checkpointed before an overlength run ends, without executing that response's
tool calls. The gym's usual cleanup cancels outstanding workers and saves their
traces. The stop reason is `sequence_limit`, not successful completion. Overshoot
by one teacher response is expected. Time and recursion limits remain in force.

Raw traces retain `sft_format`, `sequence_length`, and `stop_reason`. Each teacher
response also records its measured length. No model outputs are truncated or
deleted. Batch checking uses the same entry point:

```bash
PYTHONPATH=src:. python -m sft.filtering --tokenizer /path/to/student-tokenizer trace1.json trace2.json
```

It prints one JSON result per trace, including `tokens`, `limit`, and `keep`.
This is a length filter only; quality and completion filters remain separate.
Older traces without `sft_format` need their original system message and tool
schemas supplied to the Python API; missing context is never silently guessed.
Use the same tokenizer snapshot for collection and training. A different chat
template fails explicitly. Transformers is required in the collection environment.

WideSeek's SFT entry point requires `--student-tokenizer`; its eval entry point
does not enable the guard. Toolathlon's collection image uses the SFT-specific
graph from `sft/toolathlon_gym/langgraph.json`, which wraps the gym teacher with
the guard. Build `sft/toolathlon_gym/inference/Dockerfile.sequence-limit` over an
existing validated collection image, with tokenizer-only files in the build
context's `student-tokenizer/` directory. No model weights are needed there.

Only the teacher trajectory is counted. Worker conversations stay raw and do not
contribute unless their reports appear in the teacher's tool observations.

## Review Before Merging

The SFT owner should confirm that this rendering matches the training input:
student tokenizer and chat template, system message, tool schemas, and reasoning
removal. This change provides the shared batch API; it does not wire that API into
an existing training dataset loader.

Collection reporting classifies `sequence_limit` as **skipped**.
Such an attempt consumes an attempt slot but counts as neither an error nor a
success. Raw traces and any diagnostic native score must remain available.
Skips are terminal for resume and excluded from error rates and success counts.
