# Toolathlon-Gym RL: initial audit

Inspected 2026-09-10. This is a design and repository audit, not a runnable trainer.

## Synchronization update (2026-09-10)

The audit findings below describe the original state. Subsequently, the five local
Gym commits were merged into the RL branch. Their combined diff exactly matched
the uncommitted `gyms/` and `tests/` changes on Hertz-2. Both `toolathlon-gym` and
`we_rl_toolathlon_gym` were synchronized locally, on Hertz-2, and on GitLab/GitHub.
The remote email fixture and lockfile edits are now preserved in
`gyms/toolathlon_gym/patches/emails-mcp-fixtures.patch` and applied by the Gym
Dockerfile. Original remote changes are retained in named Git stashes; the
submodule working tree is clean. Patch application against the clean submodule
was checked and 47 focused Gym tests passed on Hertz-2. A full image build has
not yet been performed after this change.

Removed 31 dangling PostgreSQL/Kubernetes volumes and dangling image layers,
recovering approximately 27 GiB (free root disk: 26 -> 53 GiB). Models,
checkpoints, traces, active service containers and their attached volumes were
preserved. Deleted scratch databases are not recoverable; fixtures can recreate
fresh environments. Additional storage planning is still needed for RL.

## Repository state

- New branch: `we_rl_toolathlon_gym`, based on GitLab `toolathlon-gym` at `1bda75a`.
- Local `toolathlon-gym` has five additional commits ending at `d443ce5`: adaptive collection, outage backoff, database volume cleanup, coverage-first scheduling, and bounded retries. These were not silently merged into the RL branch.
- Hertz-2 `~/decomposer-qwen` is at `f391d2f` with uncommitted scheduler, runner, test, and submodule changes. Some overlap the local commits; complete equivalence is not established. Reconcile before reusing production lifecycle code.
- Existing untracked `gyms/toolathlon_gym/toolathlon.prompt.txt` and leftover `external/toolathlon/` were preserved.
- Candidate SFT model: `~/models/qwen35-4b-nonthinking-mixed-v3-8gpu` on Hertz-2 (20 GB). Its config declares `Qwen3_5ForConditionalGeneration`. Model provenance, prompt, tokenizer and export contents need checking before training.
- Root disk on Hertz-2: 99% occupied, 26 GB available at inspection. Resolve storage before installing the training stack or saving checkpoints.

## References inspected

1. [agentic-rag / occ-train](https://gitlab.2a2i.org/optimal-cognitive-core/agentic-rag/-/tree/ic/feat/train-verl09-agentloop/occ-train), commit `55bf0546fa096d881a645bdb6a70965041a15317`.
   - Current integration: `occ_train/rl/agentloop/agentic_rag_agent_loop.py`.
   - Data: `prepare_rl_data.py`; launch: `run_agentloop.sh`.
   - Pinned veRL 0.9.0 commit `483b8a009ba3a97563edee3a19887e4862b8094a` in `occ_train/rl/verl090/UPSTREAM`.
   - Two explicit patches: Qwen3.5 text-model/vLLM hybrid compatibility and LoRA checkpoint export. The model-class patch addresses a different declared architecture from our candidate checkpoint; do not copy blindly.
   - Supports several training topologies, including fully asynchronous training. This is more machinery than our initial setup requires.
2. [tau2-gym](https://gitlab.2a2i.org/Ionov/tau2-gym), commit `e9f4fc4995252cc55e8ac0128e0d2e4bdd514a72`.
   - Current integration: `training/tau2_agent_loop.py`; launch: `training/scripts/run_grpo_agentloop_4b.sh`.
   - Recipe: `training/configs/grpo/champion.yaml`; runbook: `training/README_GRPO.md`.
   - Runbook stack: veRL 0.9 + patches, torch 2.10/CUDA 12.8, vLLM 0.19, transformers 5.5.4. These are reference pins, not a validated environment on Hertz-2.
   - Recipe uses LoRA rank 32/alpha 64, group 8, batch 8, sequence-mean/token-mean loss. These are task-specific choices, not established Toolathlon optima.
   - Explicit generation timeout prevents one hung episode from holding the training batch indefinitely.

Both implement veRL's async AgentLoop interface, retain raw generated token IDs, mask observations out of policy loss, carry rollout weight-version metadata, and offload blocking environment operations. Both inspected launchers disable rollout prefix caching. The reason and compatibility with weight updates require testing; inference-only caching results do not settle this.

The [official AgentLoop documentation](https://verl.readthedocs.io/en/latest/advance/agent_loop.html) describes token IDs and response masks as the integration boundary. Its latest API can differ from the pinned reference version.

## Smallest proposed implementation

Train only the SFT Decomposer. Keep the subagent model frozen and independently served. Use ordinary GRPO with asynchronous episode execution and a batch barrier for weight updates initially. Async environment execution and fully asynchronous policy training are different choices.

Proposed files, with one clear responsibility each:

| File | Responsibility |
| --- | --- |
| `training/toolathlon_gym/README.md` | Exact setup, smoke, train, resume and export commands |
| `training/toolathlon_gym/setup.sh` | Reproducible isolated training environment and explicit required patches |
| `training/toolathlon_gym/train.sh` | One direct veRL command and resolved configuration capture |
| `training/toolathlon_gym/config.yaml` | One visible GRPO recipe, model paths, budgets and resource allocation |
| `training/toolathlon_gym/agent_loop.yaml` | Register the Decomposer AgentLoop |
| `training/toolathlon_gym/agent_loop.py` | veRL generation, raw token accounting, response masks, reward return |
| `training/toolathlon_gym/prepare_data.py` | Stable task IDs and task-level train/validation split |
| `gyms/toolathlon_gym/episode.py` | Reusable start, native score, stop lifecycle extracted from current runner |

Use veRL's existing checkpoint/resume/export facilities where verified. Add a small export helper only if the tested stack needs it. Avoid an experiment registry, MLSpace launch framework, adaptive collection scheduler or a new generic OCC framework in the initial implementation.

### Reuse and required changes

- Gym already starts an isolated PostgreSQL container, task container, network, workspace and LangGraph subagent service per episode. Preserve isolation between the multiple rollouts of one GRPO task. Audit tools that access state outside that sandbox before increasing concurrency.
- `task.prepare_task` changes process cwd and `sys.path`; keep it inside the episode process/container. Calling it concurrently in veRL threads is unsafe.
- `core.py` already defines spawn/wait state and middleware guards. Preserve multiple spawns, wait polling/report collection, rejection of parallel wait calls, and final-answer checks. Investigate a narrow model adapter into the existing graph first; if token accounting demands a manual loop, extract reusable delegation behavior instead of copying the whole agent.
- Current Gym `run.py` invokes the whole agent through LangChain, receives messages, and writes traces. That alone does not provide the raw generation tokens/log probabilities needed by veRL. Policy generation must go through veRL's server manager so updated weights are used.
- Train on Decomposer-generated tokens only. Subagent text, reports, tool results and injected user feedback have loss mask zero. Keep raw generated IDs and aligned log probabilities; avoid decode/re-encode drift or duplicated assistant tokens.
- Match the SFT checkpoint's system prompt, tool schema, chat template, non-thinking markers and EOS behavior. Current Gym explicitly selects the teacher prompt, which must not be assumed to match SFT.
- In the inspected Gym runner, an agent exception raises before native evaluation. Extract scoring so valid partial state can be evaluated after timeout, with separate agent status and evaluator status.
- Infrastructure/evaluator errors must be explicit, with bounded retries or training failure. Do not silently turn an unavailable evaluator into a task-quality reward of zero.
- Test raw native rewards and partial-score parsing before choosing binary versus fractional reward. Earlier collection's >90% acceptance threshold is not automatically an RL reward definition. Log strict pass and native partial score separately; start without speed/cost reward shaping.
- Put deadlines on generation, subagents, environment startup, evaluator and whole episode; cancellation must clean actual containers/processes, not just cancel a waiting coroutine.

## First working milestone

1. Reconcile only the lifecycle fixes needed for RL; verify SFT provenance/template. Choose storage and exact runtime pins. Do not install veRL into the working inference/evaluation environment.
2. CPU/macOS tests with fake generation and fake subagents: spawn/wait behavior, mask alignment, prompt rendering, budget boundaries, timeout/error cleanup. No CUDA or Docker needed for those tests.
3. Hertz-2: one real Gym episode, then two independently reset rollouts of the same task. Verify native reward and that one sample cannot modify another sample's database/workspace.
4. One GRPO update: provisional 2 tasks x 2 rollouts with short explicit smoke budgets. Check finite loss, nonempty policy masks, meaningful within-group reward variation, changed trainable weights, and refreshed rollout weights.
5. Save, resume for another update, export and load the result for inference. Only then run a longer experiment.

Provisional initial hardware: two H200s for veRL actor/rollout plus one for frozen subagents. This is a starting test allocation, not a measured minimum or throughput recommendation. Keep GPU ownership explicit. Use Qwen3.5-4B subagents consistent with the recent direction, subject to the first RL recipe decision.

256K inference context does not prove 256K training fits. Measure training memory at the chosen sequence budget and expose total rollout tokens (including observations), generated-token budget, turn cap and wall-clock cap separately. Temporary smoke budgets must be recorded explicitly.

All artifacts should live under `artifacts/training/toolathlon_gym/<run_id>/`: resolved config, source/image/dependency revisions, checkpoint/tokenizer identity, task split, raw model outputs, generated IDs/log probabilities/masks, subagent traces, native evaluator outputs, phase timings, checkpoints and resume state. Store credentials outside artifacts.

## Future OCC convergence

Align with the colleagues on veRL version, AgentLoop input/output contract, reward/status fields and checkpoint export format. Keep environment-specific code in separate adapters. This permits mixing datasets by `agent_name`/`data_source` later without building the shared platform now. Agree on one policy tokenizer and tool/prompt format before combining training data; a shared trainer alone does not guarantee compatible models.

No GPU jobs, package installations, training code or remote source modifications were performed for this audit.
