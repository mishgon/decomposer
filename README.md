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
uv run python examples/minimal/run.py
```

The final answer is printed, the raw state is saved to `examples/minimal/trace.json`,
and the Gantt chart is generated in `examples/minimal/trace.html`.
See `examples/minimal/README.md` for details.

## Repo structure

- `src/decomposer/`: core Decomposer package. This should stay benchmark- and training-agnostic.
- `examples/`: runnable examples of configuring and using Decomposer.
- `gyms/<gym_name>/`: reusable environment code for loading tasks, exposing tools, running a ReAct agent or Decomposer on one task or several tasks in parallel, and native result checking.
- `evals/<gym_name>/`: scripts for evaluating agents on all tasks in an environment, aggregating metrics, and saving traces for error analysis.
- `sft/<gym_name>/`: code for SFT traces collection on a gym. Shared SFT training code lives alongside these directories in `sft/`.
- `opd/<gym_name>/`: code for on-policy distillation (OPD) on a gym.
- `rl/<gym_name>/`: code for reinforcement learning (RL) on a gym.
- `external/`: third-party repositories, submodules, or vendored code.
- `tests/`: lightweight checks for reusable code and harness utilities.
- `docs/`: design notes, experiment notes, and persistent documentation.

Evaluation and training workflows reuse `gyms/<gym_name>/`.


## Tests

```bash
uvx --with . pytest
```
