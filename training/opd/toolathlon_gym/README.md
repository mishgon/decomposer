# Toolathlon Gym on-policy distillation

Teacher scoring, smoke/full configuration composition and the actual masked
loss have passed tests. On Hertz-2, `smoke-opd-20260922` completed its first
end-to-end GPU update and saved model, optimizer and training state at
`global_step_1`. This verifies the training path, not convergence or overfitting;
the original smoke stopped at update 5 on a text re-tokenization mismatch.
Teacher requests now send exact student token IDs and verify returned token
meanings, avoiding text re-tokenization of non-canonical generated sequences.

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

The preflight checks every scored token ID and its decoded meaning against
the teacher's prompt scores. A mismatch is an error, never a substituted zero.
Requests/responses are saved without authentication headers. The first token
has no context and is excluded from student response targets.

Verified on Hertz-2 on 2026-09-22: an existing Gym trajectory with 8,575 tokens
and 4,245 student-generated tokens was scored with exact token-ID alignment.
This validates that trajectory, not universal tokenizer equivalence: the same
alignment check must run for every training sample.

## Training

Use a separate checkout while RL is active. `setup.sh` applies the existing RL
dependency setup plus one optional teacher-manager hook to pinned veRL. The
hook changes no behavior unless the OPD config enables it. No teacher GPU is
allocated: the client uses lmrouter.

```bash
bash training/opd/toolathlon_gym/setup.sh
export POLICY_GPU=0 ROLLOUT_GPU=1  # choose two actually free GPUs
export MODEL_PATH=/path/to/decomposer-4b-sft
export RL_GYM_IMAGE=your-verified-gym-image-id
export RL_DATA="$PWD/artifacts/training/toolathlon_gym_opd/smoke-data"
export RL_ARTIFACTS="$PWD/artifacts/training/toolathlon_gym_opd/smoke"
export RAY_TMPDIR=/path/to/short/ray-temp
bash training/opd/toolathlon_gym/train.sh smoke
```

Monitor the latest OPD run separately from RL:

```bash
bash training/opd/toolathlon_gym/watch-opd.sh
# Add --once for a single snapshot, or --run /path/to/run for a specific run.
```

ClearML logging is inherited from the RL recipe; the task URL is printed in
`trainer.log` under project `decomposer-toolathlon-gym-opd`.

For the full pool, use `full` with new dataset/artifact directories. Smoke uses
`yf-sector-comparison` and `sf-hr-dept-budget-ppt-email`; full uses the same
346-task observed-partial-score pool as RL. Both evaluate on their training
pool: these scores measure fitting, not held-out generalization.

Both configs share the optimizer/model/harness recipe by linking `rl_recipe.yaml`
to `training/rl/toolathlon_gym/full.yaml`. They differ only in experiment name
and selected task pool. Shared settings include all-linear LoRA 32/64,
learning rate 1.5e-5, two optimizer passes, sequence-mean/token-mean aggregation,
45-minute episodes, checkpointing every update and evaluation every eight
updates with eight attempts/task. The unchanged Timur reference lives under
the RL directory. Our async one-task minibatch differs from Timur's eight-task
minibatch, as it already does in RL.

OPD uses one rollout/task/epoch, ten epochs, and two concurrent episodes.
Smoke therefore targets 20 updates. Native rewards do not enter the OPD loss;
they remain logged for evaluation. The k1 teacher/student logprob difference
feeds veRL's policy-gradient distillation objective, without GRPO group ranking.
Teacher calls are saved in `teacher_calls/`; existing episode traces,
checkpoints, TensorBoard and ClearML logging are reused.

The separate RL checkout on Hertz-2 retains its old paths deliberately;
only the new OPD checkout receives this reorganization. Do not pull the path
migration into a live RL checkout.
