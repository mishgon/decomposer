# On-policy distillation (OPD)

Distils a teacher manager into the Qwen3.5-4B Decomposer manager on the student's
own rollouts. The student samples a trajectory, the teacher scores every token the
student generated, and the student moves towards the teacher on exactly the states
it visits. Off-policy distillation (SFT on teacher traces) lives in `training/sft`;
OPD starts from an SFT checkpoint, which narrows the teacher-student gap first. That
gap is what capped tau2-gym's 27B -> 4B OPD.

The teacher is `Qwen/Qwen3.8-Flash-Next-NVFP4` on the shared LLM proxy. Token-level
OPD needs the same tokenizer as the student; `teacher.py probe` checks this, along
with token-id scoring, before anything runs.

## One round

```
checkpoint_r ──> gyms/tau2_gym/run.py --experiment opd_rollout   (student vLLM, token ids + behaviour log-probs)
             ──> samples.py     rollouts.jsonl -> samples.jsonl    (manager tokens only, episode weights)
             ──> teacher.py     samples.jsonl  -> scored.jsonl     (teacher log-prob of every student token)
             ──> train.py       one pass, torchrun + FSDP          -> checkpoint_{r+1}
             ──> (every k rounds) run.py --experiment qwen35_4b_student_checkpoint on the held-out pool
```

`loop.py` drives the rounds. Every step writes a marker under
`<run.root>/<run.name>/round_NNN/`, so an interrupted loop resumes where it stopped.
The run freezes its config into `config.yaml` and refuses to continue under a
different one. Each rollout run records a fingerprint of the checkpoint it served,
and the loop refuses rollouts that were not produced by the round's checkpoint.

```bash
source ~/.secrets/decomposer.env
.venv/bin/python -m training.opd.teacher probe --model Qwen/Qwen3.8-Flash-Next-NVFP4
.venv/bin/python -m training.opd.loop --config training/opd/configs/tau2_qwen35_4b.yaml --dry
.venv/bin/python -m training.opd.loop --config training/opd/configs/tau2_qwen35_4b.yaml
```

`tau2_qwen35_4b_smoke.yaml` runs one round of four tasks entirely on one GPU.

Qwen3.5's gated-delta backward needs Triton >= 3.7.1 on Hopper; flash-linear-attention
refuses older versions, which give wrong gradients. The project venv pins 3.6, so the
torchrun step gets the Triton 3.7.1 overlay from `train.pythonpath` and
`FLA_TILELANG=0`, as the SFT jobs do.

The smoke round (4 tasks, 2 optimizer steps, 2026-09-23) passed end to end:
- reverse KL 0.60 nats/token against Qwen3.8-Flash-Next;
- `log_ratio_mean` -0.0008, so the trainer reproduces vLLM's sampling log-probs;
- the exported checkpoint served by vLLM for the held-out evaluation.

## Where the tokens come from

Gym's vLLM model server, with `return_token_id_information: true`, stamps the last
output item of every manager turn with that turn's exact `prompt_token_ids`,
`generation_token_ids` and `generation_log_probs`. The Decomposer agent replays those
items verbatim into `rollouts.jsonl`, so nothing is re-tokenised: the trainer sees the
student's exact context and sampled ids.

Consecutive turns merge into one sequence when turn k+1's prompt extends turn k's
prompt + generation. Otherwise the chat template re-rendered the history (tool-call
arguments, for instance), and the turn becomes a sample of its own. Only generated
manager tokens carry loss; subagent reports, tool outputs and harness nudges are
context. Per-token weights make the objective the mean over episodes of the token
mean within each episode, however many samples an episode splits into.

Rollouts sample with temperature 1, no top-k/top-p and no penalties
(`UNTRUNCATED_SAMPLING`). The estimators below assume the sampled token came from
the policy itself.

## Loss

Per generated token `y_t` with behaviour log-prob `log b` (reported by vLLM while
sampling) and teacher log-prob `log T`:

```
A_t  = clamp(log T(y_t) - log b(y_t), -4, 4)          # negative per-token reverse-KL estimate, detached
r_t  = exp(log pi(y_t) - log b(y_t))
loss = sum_t w_t * -min(r_t A_t, clip(r_t, 1 +/- 0.2) A_t)
```

This is the sampled-token reverse-KL policy gradient of tau2-gym's `online_pg_opd_k1`,
with a PPO-clipped ratio against the behaviour policy. Within a round the policy
drifts from the checkpoint that sampled the data. The ratio also absorbs the
vLLM/trainer numeric mismatch, whose size `opd/log_ratio_mean` reports on the first
step. `loss.reward_coef > 0` adds a task-reward advantage (the rollout's reward minus
the mean reward of the same task in the round), for OPD plus RL.

Logits are computed only at the positions that predict trained tokens
(`logits_to_keep` with an index tensor). Log-probs are taken in fp32 chunks with
recompute, so a 64k-token context never materialises `[T, 248k]` logits. Weights stay
in fp32 with bf16 compute, and the export is fp32: with learning rates around 1e-6,
bf16 weights would round most updates away. vLLM serves the export with
`--dtype bfloat16`.

## Known limits of this driver

- Each round starts a fresh optimizer; AdamW moments do not carry across rounds.
- Every round restarts the student vLLM (~2.5 min) and runs rollout and training
  back to back on the same GPUs.
- Checkpoints are fp32 (~20 GB with the frozen vision tower) and all rounds are kept.

These are properties of the iterative driver, not of the pieces. An online trainer
(verl 0.9 driving the unchanged Decomposer Gym agent over HTTP, through Gym's
`nemo_gym` agent-loop recipe) replaces `loop.py` and keeps the rest:
- `samples.py` becomes the agent-loop output conversion;
- `teacher.py` becomes the teacher log-prob hook;
- `loss.py` becomes the custom policy loss.
