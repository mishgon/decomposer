# Decomposer

## Goal

Build Decomposer, an agent that orchestrates other agents to solve tasks faster, at lower cost, and with higher quality by:

- Parallelizing work across multiple agents.
- Routing tasks to cheaper models according to task difficulty.
- Reducing each agent's context.
- Handling subagent errors.

We aim to demonstrate improvements in speed, cost, and task-solving quality over standalone agents on Gaia2, Toolathlon, BrowseComp, and WideSearch.

## Methodology

Decomposer uses a minimal harness: a standard tool-calling loop with four tools for orchestrating subagents:

- `new(subagent_type_id) -> subagent_id`: creates a subagent of the specified type with an empty conversation history.
- `fork(subagent_id) -> subagent_id`: creates a subagent of the same type with a copy of the source conversation and state. Subsequent conversations are independent; the external environment remains shared.
- `run(subagent_id, prompt) -> subagent_run_id`: starts a run of an existing subagent and immediately returns its run ID. The same subagent can run multiple times, retaining its conversation history across runs.
- `wait() -> [...]`: waits for at least one new run to finish and returns all newly available subagent responses since the previous `wait` call. Each result includes the run status and any error. Waiting is bounded by a timeout.

This loop enables fully asynchronous orchestration and execution: Decomposer can launch newly unblocked subtasks without waiting for unrelated subagent runs to finish. It can also adapt its decomposition as subagent results arrive. See `src/decomposer/core.py` for the implementation.

We plan to train the orchestration model in two stages:

1. Off-policy distillation of a carefully prompted LLM into a smaller model.
2. Reinforcement learning that optimizes final task-solving quality, speed, and cost.

## Get started

The minimal example runs Decomposer with DeepSeek V4 Flash 0731 through OpenRouter and
Qwen3.5-4B non-thinking workers through local vLLM and LangGraph servers.
From the repository root, install the development environment and start vLLM:

```bash
uv sync
scripts/vllm/serve_qwen_3_5_4b.sh
```

With `OPENROUTER_API_KEY` set, start the subagent server in another terminal:

```bash
scripts/subagents/serve.sh
```

Then run Decomposer from the repository root:

```bash
uv run python examples/minimal/run.py
```

The final answer is printed and the complete message history is saved to
`examples/minimal/messages.md`. See `examples/minimal/README.md` for details.

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

## Development setup

Clone `decomposer` with its submodules and switch to the required branches in `decomposer` and `Gym`:

```
git clone --recurse-submodules git@github.com:mishgon/decomposer.git

git switch <decomposer-branch-name>
git -C external/Gym switch <Gym-branch-name>
```

The root project and Gym intentionally use separate environments and locks; do not combine them into a uv workspace:

```bash
# Decomposer package: root .venv and uv.lock
uv sync
uv run pytest tests

# Gym CLI: external/Gym/.venv and external/Gym/uv.lock
cd external/Gym
uv sync --extra dev
```

Gym creates another environment for `responses_api_agents/decomposer_agent` from its `requirements.txt`. It installs Gym and the root `decomposer` package in editable mode, so changes in either checkout are immediately visible without installing all Gym dependencies in the root environment.

Put dependencies imported by `src/decomposer` in the root `pyproject.toml`; put Gym-agent-only dependencies in the agent's `requirements.txt`. Commit `pyproject.toml` and `uv.lock` together.

## Tests

```bash
uvx --with . pytest
```
