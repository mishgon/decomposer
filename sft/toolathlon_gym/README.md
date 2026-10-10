# Toolathlon Gym SFT Collection

Off-policy collection for the upstream four-tool Decomposer harness. This layer
owns manifests, resume, coverage scheduling, culling and provider backoff.
It invokes the shared Gym episode runner and process supervisor.

## Launch

Load router credentials into the environment without placing keys in commands.
Confirm exact deployment IDs and perform a one-task smoke before a full run.

Build the collection image using the student tokenizer snapshot (tokenizer files
only, no model weights). The base Gym image must include the current
`gyms/toolathlon_gym/requirements.txt` dependencies.

```bash
docker build -f sft/toolathlon_gym/inference/Dockerfile.refresh \
  --build-arg RUNTIME_IMAGE=YOUR_VALIDATED_GYM_IMAGE_ID \
  -t decomposer-toolathlon-sft:latest .
```

Place that snapshot in `student-tokenizer/` in the build context. The build checks
that it loads offline. The collector defaults to this SFT image and rejects plain
Gym images, including on resume, so it cannot silently omit early stopping.

```bash
PYTHONPATH=src:. python -m sft.toolathlon_gym.run \
  --all --adaptive -n 1 --concurrency 8
```

Set credentials for the models configured in the Gym image: `LLM_PROXY_MASTER_KEY`
for lmrouter or `OPENROUTER_API_KEY` for OpenRouter. Local vLLM requires neither key.
The available models are defined in [models.py](../../src/decomposer/models.py).
Models use their registered endpoints; collection does not start a local GPU server.
The Gym image refresh remains under `inference/`.

Models are configured in [Gym agents.py](../../gyms/toolathlon_gym/agents.py).
Rebuild the Gym image after changing them. For `vllm/` models, start the local
server with a container-accessible bind address before collection
(`host="0.0.0.0"` when using `vllm_server`). `collect_hosted.sh` loads credentials from
`LMROUTER_ENV` (default: `~/.local/share/environment/lmrouter.env`) and requires
a validated `COLLECTION_IMAGE`. Example:

```bash
bash sft/toolathlon_gym/collect_hosted.sh --all --adaptive -n 1 --concurrency 8
```

The two model configurations use native tool calling. The runner has no automatic
provider fallback. Old traces keep their recorded settings; start a new run when
changing generation profiles to keep collections comparable.

`inference/Dockerfile.refresh` can refresh application code on a pinned,
previously validated Python 3.12 runtime image without downloading dependencies.
It preserves that image's system packages and task-service patches. Record both
the base image ID and the resulting image ID when using it.

After each teacher response, shared `sft/sequence_limit.py` counts the student
sequence using `sft/filtering.py`: system prompt, tools, messages and observations,
without teacher reasoning or a generation prompt. Above 16,384 tokens it stops
before executing further tools. Raw reasoning and traces are retained. The result
is `skipped` with `stop_reason=sequence_limit`: it consumes an attempt slot but
counts as neither an error nor a success. Native scores remain diagnostic.

The same filter checks individual traces or batches before SFT:

```bash
PYTHONPATH=src:. python -m sft.filtering --tokenizer student-tokenizer trace1.json trace2.json
```

Use the same tokenizer snapshot as training. New traces save the prompt and tool
schemas as `sft_format`; historical traces without them need those inputs supplied
explicitly through the Python API. This branch does not change the training code
maintained on the separate `sft` branch.

### Integration Into SFT Training

The inspected SFT revision (`83f1bf34`, retained as `origin/sft` locally) has
`training/sft/train.py::_tokenization_stats`, which receives a prepared row with
`messages`, `tools` and `chat_template_kwargs`. That branch is no longer advertised
by GitLab; confirm the equivalent location in the trainer's current checkout.
Replace only its `tokenizer.apply_chat_template(...)` call with:

```python
from sft.filtering import tokenize_student

encoded = tokenize_student(
    example,
    tokenizer=tokenizer,
    training_template=training_template,
    return_assistant_tokens_mask=True,
)
```

Keep its assistant-mask validation, `_token_length = len(encoded['input_ids'])`,
dataset statistics and `_apply_overlength_policy` unchanged. Set training's
`max_length` to `32768` and enable `exclude_overlength`. Exactly 32,768 tokens
are allowed; longer examples are excluded rather than truncated.

Both collection and this replacement use the same tokenization function.
Training retains control of message preparation, reasoning settings and its
assistant-mask template. Matching counts also requires the same student
tokenizer, system prompt, tool schemas and rendered messages. A different
training template or preprocessing can change lengths; verify token-ID equality
on a saved example before treating the collection count as the training count.
The raw-trace CLI deliberately checks the recorded template hash, while the
prepared-row function accepts the trainer's explicit template.

## Coverage Policy

Each wave launches one attempt per task with no qualifying trace. A native pass
or partial score strictly above 0.90 qualifies, provided the agent finished.
Scores from agent errors/timeouts are diagnostic only. Check-marker logs without
a successful evaluator exit or explicit totals remain unscored.
After six unsuccessful launches, a task is culled, including tasks with
unparseable evaluations. Missing scores remain unknown, not fabricated zeros.
Once coverage is exhausted, the scheduler balances retained tasks toward four
qualifying traces, with at most 24 additional attempts per task in that phase.
The target is not guaranteed: the run stops when the budgets are exhausted.

Use `--adaptive-target-successes 1` for coverage only. Existing CLI threshold and
target overrides remain available. Failed traces and all native outputs are kept.

## Artifacts and Resume

The default root is `artifacts/sft/toolathlon_gym`:

- `runs/<run-id>/manifest.json`, event log and per-attempt stdout/stderr.
- `traces/<task>/<episode>/`: raw traces, model logs and task workspaces.
- `evals/<task>/<episode>/result.json`: native scores used by scheduling.

```bash
PYTHONPATH=src:. python -m sft.toolathlon_gym.run --resume RUN_ID --adaptive
```

For an existing artifact root, also pass `--gym-artifacts-dir PATH`.
Completed attempts are not repeated on resume. Do not mix pre-migration
`spawn_agent` traces and new-harness traces under a new collection identity.

A run lock rejects a second collector for the same run ID. After a crash, resume
recovers saved `attempt.json` records before scheduling work. Do not delete the
lock file: the OS releases its lock when the collector exits.
