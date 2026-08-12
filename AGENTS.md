# AGENTS.md

## Project goal

The goal of this project is to build an agent, called Decomposer, that solves tasks exclusively by decomposing them into subtasks and delegating them to subagents. Decomposer is based on a small language model trained with RL to optimize the quality, speed and cost of the whole system.

## Methodology

At a high level, idea is similar to Sakana Fugu (https://arxiv.org/abs/2606.21228, check it out for context). However, there are substantial differences. Their Conductor model does not work in a ReAct loop. It produces a static decomposition once, subagents complete the subtasks and the last subagent's response is returned as the output. Our Decomposer agent works in a standard tool-calling loop with two tools: `spawn_subagent` (spawns a new subagent and delegates a subtask to it) and `wait` (waits for subagents' reports). This enables a dynamic, adaptive decomposition. Carefully inspect our design choices and implementation details in `src/decomposer/core.py` and `src/decomposer/core.py`. Their understanding is essential for our future work.

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

Subagent types: Gemma-4-E2B / -E4B / -12B / -26B-A4B (thinking / non-thinking), 8 types in total. We select Gemma-4 models family, because they are incredibly fast and laconic compared to Qwen models. In the first version we use only Gemma-4-26B-A4B non-thinking model.

The current plan:
- [ ] Start generating SFT data and training SFT models (version and backup them on CDS and main NFS) on training envs.
- [ ] Start evaluating SFT models on test envs.
- [ ] Start setting up RL training on training envs.

Ideas:
- Train Decomposer to anonymize prompts based on [PII public data](https://huggingface.co/datasets/Pritesh-2711/pii-bench)

## Repo structure

- `src/decomposer/`: core Decomposer package. This should stay benchmark- and training-agnostic.
- `examples/`: runnable examples of configuring and using Decomposer.
- `gyms/`: environment integrations that collect traces and run native evaluation.
- `training/`: training and finetuning workflows.
- `artifacts/data/`: collected trajectories and episode workspaces ignored by git.
- `artifacts/evals/`: evaluation results and aggregate metrics ignored by git.
- `artifacts/training/`: model checkpoints and training logs ignored by git.
- `external/`: third-party repositories, submodules, or vendored code.
- `tests/`: lightweight checks for reusable code and harness utilities.
- `docs/`: design notes, experiment notes, and persistent documentation.
