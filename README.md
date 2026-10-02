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

We plan to train the orchestration model in two stages:

1. Off-policy distillation of a carefully prompted LLM into a smaller model.
2. Reinforcement learning that optimizes final task-solving quality, speed, and cost.

## Get started

The minimal example runs Decomposer with Flash Next non-thinking and Qwen3.5-4B
unlooped thinking workers through lmrouter and a local LangGraph server. Both use
`create_model(model)` in `src/decomposer/models.py`.
From the repository root, install the development environment:

```bash
uv sync
```

With `LLM_PROXY_MASTER_KEY` set, run the example. It starts a local LangGraph
server hosting Decomposer and its agents, then stops it when the run finishes:

```bash
uv run python examples/minimal/run.py
```

The final answer is printed and the complete message history is saved to
`examples/minimal/messages.md`. See `examples/minimal/README.md` for details.

## Hosted Models and Private lmrouter Access

Models and sampling settings are configured in [models.py](src/decomposer/models.py).
For private lmrouter access, ask Codex to follow
[the setup instructions in AGENTS.md](AGENTS.md#private-lmrouter-access).

### Use the Registry Anywhere

The same connection works with synchronous `invoke()` and asynchronous
`ainvoke()`. No per-gym model setup is required:

```python
from langchain.agents import create_agent
from decomposer.models import create_model

agent = create_agent(model=create_model("qwen_3_8_flash_next_non_thinking"), tools=[])
result = agent.invoke({"messages": [{"role": "user", "content": "Say hello."}]})
print(result["messages"][-1].content)
```

Toolathlon Gym automatically mounts the socket directory into task containers
and passes the key by environment variable name. Other container launchers need
the same read-only directory mount and the container-side `LLM_PROXY_UNIX_SOCKET`
path. Load the same key into the container environment without putting it in
command arguments. Keep the HTTPS registry URL unchanged.

See [Gym usage](gyms/toolathlon_gym/README.md) and
[SFT collection](sft/toolathlon_gym/README.md) for their launch commands.

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
