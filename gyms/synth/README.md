# Deterministic File-Swap Gym

Each task asks Decomposer to swap two independent pairs of files, with two
empty scratch files. Only opaque filenames and their presentation vary.
Training and held-out names come from separate fixed seeds. Contents and worker
behavior stay fixed. The policy uses the unchanged main Decomposer harness.

Workers are LangGraph graphs with **no LLM**. Their regex parser accepts one
command per line:

```text
COPY source.txt target.txt
CHECK target.txt original_source.txt
```

Actual names are eight lowercase letters followed by `.txt`. COPY reads current
contents; CHECK compares against initial contents without modifying anything.
A prompt must consist entirely of valid commands referring to the task's six
files. Invalid prompts are rejected before mutation. A batch runs in listed
order under a workspace lock. File contents live in `workspace.json`, not real
user files. No shell execution, external tools, network retrieval or LLM judge.

Two three-copy batches solve a task with six copies. Deterministic tests check
that plan through the real core and worker server. `reference_plans()` is test
code support, never injected into student or teacher prompts.

## Metrics

- Native score: correctly swapped destination files / 4.
- Pass: all four destinations correct and the policy finishes normally.
- Parallel: at least two COPY-containing worker runs launched before collection
  by `wait`. This is **logical delegation concurrency**, not CPU utilization.
- Copy count and invalid-command count: diagnostics, not reward penalties.

Correctness, parallel delegation and efficiency are reported separately. Workers
perform copies almost instantly, so wall-clock speed is dominated by policy
inference and harness polling. No artificial worker delays are introduced.
The shared harness still creates UUIDs and records timings; worker reports contain
neither. Deterministic workers do not make sampled policy trajectories deterministic.

## Run

With a local stock Qwen server on registry port 8024:

```bash
PYTHONPATH=src:. python -m gyms.synth.run --split eval --index 0
```

See [baseline evaluation](../../evals/synth/README.md) and
[OPD experiment](../../opd/synth/README.md). Both use `run.episode()` and write
task descriptions, workspace mutations, worker checkpoints, policy traces,
HTML visualization, token usage and exact grader outputs under the supplied
artifact directory. No existing artifacts are overwritten.
