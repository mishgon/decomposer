# Toolathlon Gym on-policy distillation

Work in progress: teacher scoring is implemented and integration-tested; the
training entry point and smoke/full configs are not implemented yet.

Student: decomposer-4b SFT. Subagents: hosted Qwen3.5-4B non-thinking.
Teacher: hosted `Qwen/Qwen3.8-Flash-Next-NVFP4`.

The student executes the existing Gym environment. The teacher scores its
generated token sequence; only decomposer-generated positions contribute to
the distillation loss. Tool observations and subagent reports remain context,
not training targets. Native evaluation remains the quality metric.

## Teacher preflight

Set `OPD_TEACHER_URL` to the OpenAI-compatible `/v1` endpoint,
`OPD_TEACHER_MODEL` to the teacher ID, and `OPD_TEACHER_API_KEY` (or the existing
`LLM_PROXY_MASTER_KEY`) privately in the environment. Optional
`OPD_TEACHER_HOST=hostname:address` overrides DNS while preserving TLS verification.

```bash
python -m training.opd.toolathlon_gym.teacher \
  --trace /path/to/episode/trace.json \
  --tokenizer /path/to/decomposer-4b-sft \
  --output artifacts/training/toolathlon_gym_opd/preflight/teacher-score.json
```

The preflight checks round-trip tokenization and every scored token ID against
the teacher's prompt scores. A mismatch is an error, never a substituted zero.
Requests/responses are saved without authentication headers. The first token
has no context and is excluded from student response targets.

Verified on Hertz-2 on 2026-09-22: an existing Gym trajectory with 8,575 tokens
and 4,245 student-generated tokens was scored with exact token-ID alignment.
This validates that trajectory, not universal tokenizer equivalence: the same
alignment check must run for every training sample.

The running RL experiment still uses `training/toolathlon_gym`. Its migration
to `training/rl/toolathlon_gym` must not disturb active processes. Keep the
unmodified Timur reference and shared optimization choices (LoRA 32/64,
learning rate 1.5e-5, sequence-mean/token-mean aggregation) when adding configs.
