# Decomposer

## Goal

The goal of this project is to build an agent, called Decomposer, that solves tasks exclusively by decomposing them into subtasks and delegating them to subagents. Decomposer is based on a small language model trained with RL to optimize the quality, cost and speed of the whole system. Big dream is to build a real product: Decomposer orchestrating a pull of subagents based on [frontier open-weight models optimized for local usage](https://unsloth.ai/docs/models).

## Methodology

At a high level, idea is similar to [Sakana Fugu](https://arxiv.org/abs/2606.21228), however, there are substantial differences. Sakana's Conductor model does not work in a ReAct loop. It produces a static decomposition once, subagents complete the subtasks and the last subagent response is returned as the output. On the contrary, our Decomposer agent works in a standard tool-calling loop with two tools: `spawn_subagent` (spawns a new subagent and delegates a subtask to it) and `wait` (waits for subagents' reports). This enables a dynamic, adaptive decomposition. You could check out our design choices and implementation details in `src/decomposer/core.py` and `src/decomposer/core.py`.

## Plan

The current plan is:
- Use NeMo-Gym (installed as a git submodule under `external/Gym`) as a framework for Decomposer's evaluation and traces collection on different environments. See our integration of the Decomposer agent into NeMo-Gym framework in `external/Gym/responses_api_agents/decomposer_agent`.
- Run Decomposer agent based on Qwen3.6-27B with subagents based on Qwen3.5-4B, Gemma-4-E4B, and LFM2.5-8B-A1B on several environments.
- Evaluate on the *train* splits in comparison to individual baselines based on Qwen3.5-4B, Gemma-4-E4B, and LFM2.5-8B-A1B. Work on the system prompt and few-shot examples in order to achieve reasonable traces and quality compared to baselines.
- Collect Qwen3.6-27B-based Decomposer agent traces and run SFT of Qwen3.5-0.8B / 2B / 4B on them. Evaluate resulting Decomposer-0.8B / 2B / 4B as well as the Qwen3.6-27B-based Decomposer on *test* splits.
- Further train Decomposer-0.8B / 2B / 4B with RL and compare with SFT checkpoints.
- When the first prototype with Decomposer-0.8B / 2B / 4B and Qwen3.5-4B-, Gemma-4-E4B-, and LFM2.5-8B-A1B-based subagents is proven to work, scale it to larger models.

## Repo structure

- `src/decomposer/`: core Decomposer package. This should stay benchmark- and training-agnostic.
- `evals/`: evaluation runners and benchmark-specific adapters.
- `training/`: training and finetuning workflows.
- `data/`: source code for preparing datasets used by training or evals.
- `artifacts/`: generated outputs, ignored by git.
- `external/`: third-party repositories, submodules, or vendored code.
- `tests/`: lightweight checks for reusable code and harness utilities.
- `docs/`: design notes, experiment notes, and persistent documentation.

## Development setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). The root project and Gym intentionally use separate environments and locks; do not combine them into a uv workspace.

```bash
git submodule update --init --recursive

# Decomposer package: root .venv and uv.lock
uv sync
uv run pytest tests

# Gym CLI: external/Gym/.venv and external/Gym/uv.lock
cd external/Gym
uv sync --extra dev
```

Gym creates another environment for `responses_api_agents/decomposer_agent` from its `requirements.txt`. It installs Gym and the root `decomposer` package in editable mode, so changes in either checkout are immediately visible without installing all Gym dependencies in the root environment.

Put dependencies imported by `src/decomposer` in the root `pyproject.toml`; put Gym-agent-only dependencies in the agent's `requirements.txt`. Commit `pyproject.toml` and `uv.lock` together.
