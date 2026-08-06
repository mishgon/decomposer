# Decomposer

## Goal

The goal of this project is to build an agent, called Decomposer, that solves tasks exclusively by decomposing them into subtasks and delegating them to subagents. Decomposer is based on a small language model trained with RL to optimize the quality, cost and speed of the whole system.

## Methodology

At a high level, idea is similar to [Sakana Fugu](https://arxiv.org/abs/2606.21228), however, there are substantial differences. Sakana's Conductor model does not work in a ReAct loop. It produces a static decomposition once, subagents complete the subtasks and the last subagent's response is returned as the output. On the contrary, our Decomposer agent works in a standard tool-calling loop with two tools: `spawn_subagent` (spawns a new subagent and delegates a subtask to it) and `wait` (waits for subagents' reports). This enables a dynamic, adaptive decomposition. You could check out our design choices and implementation details in `src/decomposer/core.py` and `src/decomposer/core.py`.

## Get started

The minimal example runs Decomposer with GLM-5.2 through OpenRouter and one
Gemma-4-E4B-IT subagent through a local vLLM and LangGraph server. From
the repository root, install the development environment and start vLLM:

```bash
uv sync
scripts/vllm/serve_gemma_4_e4b.sh
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

## Plan

Training envs:
- [NeMo-Gym's Workplace assistant](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/workplace_assistant)
- [Toolathlon-Gym](https://github.com/eigent-ai/toolathlon_gym)
- [NeMo-Gym's Finance Sec Search](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/finance_sec_search)
- [NeMo-Gym's Google Search](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/google_search)
- [Z.ai's DeepDive](https://huggingface.co/datasets/zai-org/DeepDive)
- [WideSeek-R1](https://huggingface.co/collections/RLinf/wideseek-r1)

Test envs:
- [Gaia2](https://huggingface.co/datasets/meta-agents-research-environments/gaia2)
- [Toolathlon](https://github.com/hkust-nlp/Toolathlon)
- [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)
- [GPQA Diamond](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/gpqa_diamond)

Subagent types: Gemma-4-E2B / -E4B / -12B / -26B-A4B (thinking / non-thinking), 8 types in total. We select Gemma-4 models family, because they are incredibly fast and laconic compared to Qwen models.

The current plan:
- [ ] Establish the system prompt, few-shot examples and teacher model(s) ([GLM-5.2](https://openrouter.ai/z-ai/glm-5.2), [GPT-5.6 Luna](https://openrouter.ai/openai/gpt-5.6-luna), [deepseek-4-flash](https://openrouter.ai/deepseek/deepseek-v4-flash-20260731)) by comparing their quality, speed and subagent types use on training envs.
- [ ] Start generating SFT data and training SFT models (version and backup them on CDS and main NFS) on training envs.
- [ ] Start evaluating SFT models on test envs.
- [ ] Start setting up RL training on training envs.

Ideas:
- Train Decomposer to anonymize prompts based on [PII public data](https://huggingface.co/datasets/Pritesh-2711/pii-bench)

## Repo structure

- `src/decomposer/`: core Decomposer package. This should stay benchmark- and training-agnostic.
- `examples/`: runnable examples of configuring and using Decomposer.
- `evals/`: evaluation runners and benchmark-specific adapters.
- `training/`: training and finetuning workflows.
- `data/`: source code for preparing datasets used by training or evals.
- `artifacts/`: downloaded data, saved model checkpoints, training logs and evaluation metrics ignored by git.
- `external/`: third-party repositories, submodules, or vendored code.
- `tests/`: lightweight checks for reusable code and harness utilities.
- `docs/`: design notes, experiment notes, and persistent documentation.

## Development setup

Clone `decomposer` repo with `Gym` as submodule and switch to the required branches in both `decomposer` and `Gym`:

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
