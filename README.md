# Decomposer

## Goal

Build Decomposer, an agent that orchestrates other agents to solve tasks faster, at lower cost, and with higher quality by:

- Parallelizing work across multiple agents.
- Routing tasks to cheaper models according to task difficulty.
- Reducing each agent's context.
- Handling agent errors.

We aim to demonstrate improvements in speed, cost, and task-solving quality over standalone agents on Gaia2, Toolathlon, BrowseComp, and WideSearch.

## Methodology

Decomposer uses a minimal harness: a standard tool-calling loop with four tools for orchestrating agents:

- `new(agent_type_id) -> agent_id`: creates an agent of the specified type with an empty conversation history.
- `fork(agent_id) -> agent_id`: creates an agent of the same type with a copy of the source conversation and state. Subsequent conversations are independent; the external environment remains shared.
- `run(agent_id, prompt) -> agent_run_id`: starts a run of an existing agent and immediately returns its run ID. The same agent can run multiple times, retaining its conversation history across runs.
- `wait() -> [...]`: waits for at least one new run to finish and returns all newly available agent responses since the previous `wait` call. Each result includes the run status and any error. Waiting is bounded by a timeout.

This loop enables fully asynchronous orchestration and execution: Decomposer can launch newly unblocked subtasks without waiting for unrelated agent runs to finish. It can also adapt its decomposition as agent results arrive. See `src/decomposer/core.py` for the implementation.

We plan to train the orchestration model in three stages:

1. Off-policy distillation of a carefully prompted LLM into a smaller model.
2. On-policy distillation using the same teacher.
3. Reinforcement learning that optimizes final task-solving quality, speed, and cost.

## Get started

The minimal example runs Decomposer with Flash Next non-thinking and Qwen3.5-4B
unlooped thinking workers through lmrouter and a local LangGraph server. Both use
`create_model(model_id)` in `src/decomposer/models.py`.
From the repository root, install the development environment:

```bash
uv sync
```

With `LLM_PROXY_MASTER_KEY` set, run the example. It starts a local LangGraph
server hosting Decomposer and its agents, then stops it when the run finishes:

```bash
uv run python -m examples.minimal.run
```

The final answer is printed, the raw state is saved to `examples/minimal/trace.json`,
and the Gantt chart is generated in `examples/minimal/trace.html`.
See `examples/minimal/README.md` for details.

## Repo structure

- `src/decomposer/`: core Decomposer package. This should stay benchmark- and training-agnostic.
- `examples/`: runnable examples of configuring and using Decomposer.
- `gyms/<gym_name>/`: environment code for loading tasks, exposing tools, running Decomposer or another agent on all or some tasks, trace generation, and saving raw results.
- `evals/<gym_name>/`: one-command evaluation on top of `gyms/<gym_name>` (run, then compute metrics), plus comparisons and trace statistics.
- `sft/`: shared SFT code (dataset schema, builder and adapters; trainer, templates and configs). `sft/<gym_name>/` holds gym-specific trace preparation and release specs.
- `opd/`: shared on-policy distillation code; `opd/<gym_name>/` holds per-gym loop configs.
- `rl/<gym_name>/`: reinforcement learning on a gym (empty for now).
- `artifacts/data/`: collected trajectories and episode workspaces ignored by git.
- `artifacts/evals/`: evaluation results and aggregate metrics ignored by git.
- `artifacts/training/`: model checkpoints and training logs ignored by git.
- `external/`: third-party repositories, submodules, or vendored code.
- `tests/`: lightweight checks for reusable code and harness utilities.
- `docs/`: design notes, experiment notes, and persistent documentation.

Evaluation and training workflows reuse `gyms/<gym_name>/`.


## Tests

```bash
uvx --with . pytest
```

## Accelerated Qwen3.5 SFT

The current four-H100 Qwen3.5 recipe uses batch 2 per GPU, length grouping,
FLA plus causal-conv1d for linear-attention layers, and a pinned HF Hub
FlashAttention-2 kernel for full-attention layers. Normal training jobs do not
need NVCC; it is needed only once to build the reusable causal-conv1d bundle.

See [the Qwen3.5 fast-runtime guide](docs/sft_qwen35_fast_runtime.md) for exact
environment preparation, local smoke, MLSpace dry-run/submission commands,
artifact paths, and troubleshooting.
