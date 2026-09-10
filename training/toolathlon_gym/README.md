# Toolathlon-Gym RL

Working branch: `we_rl_toolathlon_gym`. Remote checkout: `/home/matrosov/decomposer-rl`.
This is the integration in progress, **not yet a runnable veRL trainer**.
Existing collection and benchmark checkouts remain separate.

## First piece: reproducible task selection

Run from the repository root using ordinary Python; no CUDA dependencies:

```bash
python3 -m unittest discover -s tests -p test_toolathlon_gym_rl_data.py
python3 training/toolathlon_gym/prepare_data.py \
  --output artifacts/training/toolathlon_gym/task-split.json
```

The manifest includes every Gym task, its config hash and the Gym source revision.
It refuses to overwrite an existing manifest. `--validation-tasks N` reserves an
explicit, deterministic task-level holdout; the default reserves none. Repeated
rollouts must inherit their task's split. This is a task manifest, not veRL's final
prompt/parquet dataset. It does not assert that every task's infrastructure works.

## Next integration steps

1. Extract isolated episode start/score/stop from the Gym runner; test cancellation
   and scoring partial state without silently rewarding infrastructure failures.
2. Connect existing Decomposer delegation to veRL's generation interface, retaining
   generated token IDs/logprobs and masking observations out of policy loss.
3. Build a separate, pinned training environment aligned with the colleagues'
   veRL 0.9 AgentLoop implementations. Do not install into the evaluation venv.
4. Test two isolated episodes, one GRPO update, then checkpoint save/resume/export.

No GPU allocation is made by task preparation. The model paths remain under
`/home/matrosov/models`; do not duplicate weights here. Put new training artifacts
under `artifacts/training/toolathlon_gym/<run_id>/`.
See [the integration audit](../../docs/toolathlon_gym_verl_plan.md) for the design
and reference revisions.
