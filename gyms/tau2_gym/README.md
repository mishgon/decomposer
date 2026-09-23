# tau2 gym

Runs the Decomposer on Timur's tau2 GAIA2-hard domains, through NeMo-Gym.

The environment is exposed as a **NeMo-Gym resources server hosted outside the
`external/Gym` submodule**. `gym env start` finds it because `NEMO_GYM_EXTRA_ROOTS`
points at `gym_components/`, and extra roots are searched before Gym's own tree
(`nemo_gym/__init__.py:52-79`). No submodule edits are required.

```
gyms/tau2_gym/
├── experiments.py            # the experiment registry: manager backend, model, prompt, sampling
├── run.py                    # local runner: manager + subagents -> langgraph -> gym env -> gym eval -> validate
├── tau2_export.py            # builds the Gym dataset from a task pool (runs inside the tau2 venv)
├── task_pools/               # versioned task pools and the recipe-driven builder
├── analyze_traces.py         # decomposition / parallelism statistics
├── subagents/                # LangGraph subagent server (Qwen3.5-4B, non-thinking)
└── gym_components/           # <- NEMO_GYM_EXTRA_ROOTS
    ├── pyproject.toml        # marker; see "Why the marker file" below
    └── resources_servers/tau2_gym/
        ├── app.py            # seed_session / POST {tool} / verify
        ├── tau2_bridge.py    # all tau2 imports: tasks, envs, schemas, scoring
        └── requirements.txt
```

## One-time setup

The submodule is `external/tau2_gym`. Hertz-2 has no GitLab access, so its submodule
fetches from the `~/tau2-gym` checkout there, which is fast-forwarded from bundles made
on a machine that has GitLab:

```bash
# laptop: only the commits Hertz-2 lacks (the full history carries ~600 MB of data/sft)
git -C tau2-gym bundle create tau2-gym.bundle <hertz-2 commit>..main
scp tau2-gym.bundle Hertz-2:repo-bundles/
# Hertz-2
git -C ~/tau2-gym fetch ~/repo-bundles/tau2-gym.bundle main && git -C ~/tau2-gym merge --ff-only FETCH_HEAD
# once: point the submodule at that checkout (local config, not committed)
git config submodule.external/tau2_gym.url ~/tau2-gym
git -C external/tau2_gym remote set-url origin ~/tau2-gym
# after every pin change
git -C external/tau2_gym fetch origin && git submodule update external/tau2_gym
```

Create the tau2 venv used by `tau2_export.py` and the integration tests (tau2's
dependency set is heavy, so it is deliberately kept out of the project venv):

```bash
source ~/proxy_on.sh
uv venv --python 3.12 /home/sukhorukov/decomposer_artifacts_new/venvs/tau2
VIRTUAL_ENV=/home/sukhorukov/decomposer_artifacts_new/venvs/tau2 \
  uv pip install -e external/tau2_gym pytest
```

## Task pools

A pool is `task_pools/<name>.json` (`{domain: [task_id, ...]}`) plus a `.meta.json`
recording its recipe, counts, the tau2 commit and a sha256. Runs and SFT sources refer
to pools by name, and `load_pool` rejects a pool edited after it was built.

| pool | role | tasks / domains | recipe |
|---|---|---|---|
| `decomposer_pool_v1` | train | 333 / 17 | canon ∪ dead − held-out, ScriptedUser domains only |
| `decomposer_train_v2` | train | 752 / 40 | canon ∪ dead − held-out, every canon domain |
| `decomposer_eval_v1` | eval | 455 / 16 | HELDOUT_v2 − QUARANTINE: domains reserved from all tau2 training |

"canon" is `canon_full_{sft,grpo,dpo}.json`; "dead" are tasks with zero passes in
tau2's solo-4B calibration (`progress/gaia2/unified/full/full_report_*.json`); "held-out"
is `HELDOUT_v2`, `heldout_exec_v5`, `cycle_heldout` and `QUARANTINE`, plus every domain
HELDOUT_v2 reserves. v1 kept only `_user_type_for_domain(d) == "scripted"` domains; v2
drops that filter because the Decomposer only ever sends the task's `first_message`
and tau2's own training forces ScriptedUser on every domain.

To grow the pool, add a recipe to `build_pool.py` under a new name (never edit a built
pool) and build it; the builder refuses any train/eval task overlap:

```bash
.venv/bin/python gyms/tau2_gym/task_pools/build_pool.py --list
.venv/bin/python gyms/tau2_gym/task_pools/build_pool.py decomposer_train_v3
```

## Experiments

`experiments.py` is the only description of an experiment; `run.py` generates the Gym
config from it and writes it to `<run>/configuration/tau2_gym.yaml`.

| experiment | manager | prompt | pool |
|---|---|---|---|
| `qwen38_flash_teacher_{non_thinking,thinking}` | Qwen3.8-Flash-Next via the LLM proxy | teacher | train_v2 |
| `deepseek_v4_flash_teacher` | DeepSeek-v4-flash via OpenRouter (effort max) | teacher | train_v2 |
| `qwen35_4b_base_student` | untuned Qwen3.5-4B, local vLLM | student | eval_v1 |
| `qwen35_4b_sft_mixed_v3_student` | Qwen3.5-4B SFT (v5 student release), local vLLM | student | eval_v1 |
| `opd_rollout` | the OPD round's checkpoint, local vLLM, untruncated sampling, token ids + logprobs | student | train_v2 |

Every experiment exposes one subagent type, `subagent_non_thinking`, with the SFT
releases' canonical description, so teacher traces, SFT data, student evals and OPD
rollouts all see the same `spawn_subagent` schema.

Remote Qwen managers go through `gyms.remote_model_proxy` on port 8144 with the
`qwen3_xml` normaliser and the experiment's sampling as `--extra-body-json`. Gym's
manager speaks the Responses API, where the shared proxy **ignores
`chat_template_kwargs`**: thinking is switched with `reasoning.effort` (`none` or the
default `xhigh`). Before this was found, a "non-thinking" manager silently reasoned at
xhigh.

## Run

```bash
source ~/.secrets/decomposer.env      # LLM_PROXY_* (and OPENROUTER_API_KEY_DECOMPOSER)

# what would run, including the generated Gym config
.venv/bin/python gyms/tau2_gym/run.py --experiment qwen38_flash_teacher_non_thinking --dry

# teacher traces for SFT: no GPU at all (manager and subagents on the proxy)
.venv/bin/python gyms/tau2_gym/run.py --experiment qwen38_flash_teacher_non_thinking --num-repeats 4

# a student checkpoint on the held-out pool, one domain-stratified task per domain
.venv/bin/python gyms/tau2_gym/run.py --experiment qwen35_4b_sft_mixed_v3_student \
  --tasks-per-domain 5 --num-repeats 3 --manager-gpu 6

# OPD rollouts from a round's checkpoint (what training/opd drives)
.venv/bin/python gyms/tau2_gym/run.py --experiment opd_rollout --manager-checkpoint <dir> \
  --manager-gpu 5 --output-dir <round>/rollouts
```

`--pool` overrides the experiment's pool and `--tasks-per-domain k` takes a
deterministic, hash-stratified subsample. The run name records experiment, pool,
subsample and repeats. `run_status.json` records the experiment, pool sha, generated
config sha and, for local managers, a fingerprint of the served checkpoint.

## Subagent backends

`--subagent-backend` picks where subagent traffic goes. The subagent graph is identical
either way: it always talks plaintext HTTP to loopback with `api_key="EMPTY"`, resolving
its endpoint from `TAU2_GYM_MODEL_BASE_URLS_JSON`.

| | `llm_proxy` (default) | `local_vllm` |
|---|---|---|
| Model host | shared Qwen3.5-4B replicas | dedicated vLLM this runner starts |
| GPU | **none** | one (`--subagent-gpu`), held for the whole run |
| `model_startup` | **~2 s** | 140-360 s |
| Needs | `LLM_PROXY_URL`, `LLM_PROXY_MASTER_KEY` | a free GPU + port |

The subagent proxy (port 8143) gets neither `--response-tool-parser` (its Qwen XML
normalisation only runs on the Responses path, `remote_model_proxy.py:317`, and
subagents use Chat Completions) nor `--extra-body-json` (it merges server-side and
*overrides caller keys*, `remote_model_proxy.py:49-54`), so the graph's own sampling is
what runs.

Measured on 3 domains x 5 tasks x 3 repeats (structural reward), the backends agree
within noise:

| | local_vllm | llm_proxy |
|---|---|---|
| pass rate | 0.756 | 0.800 |
| spawns/rollout | 1.58 | 1.64 |
| max fan-out (mean / max) | 1.33 / 4 | 1.20 / 4 |

## How scoring works

`verify` receives a *flattened* trajectory: every subagent `function_call` in report
order plus the manager's final assistant message, and **no `function_call_output`
items** (`decomposer_agent/app.py:287-311`).

`tau2_bridge.score_trajectory` turns that into the trajectory of a *virtual flat
agent*: the calls are replayed into a freshly seeded environment and every recorded
result is appended as a `ToolMessage`; the manager's final report becomes the agent's
final message. That is exactly what tau2's own `training/reward.compute_reward`
consumes, so the Decomposer is graded by tau2's binary predicate with its tri-state
terms (`None` = does not apply to this task):

- **binding**: `action`, `param`, `db`, `answer`, `env_assertion`, `restraint`,
  `side_effects`. Reward = their conjunction.
- **reported, not binding**: `tool_validity`, `signal`, `closed`. They describe a flat
  agent's own tool names and text channel, i.e. subagent formatting and the harness.

Things to know when reading a breakdown:

- `restraint` is armed with `REWARD_FORBIDDEN_PENALTY=1.0`, as in tau2's training runs.
  `tau2_bridge` sets it before importing tau2 and refuses to start if it is 0.
- `db` is `None` on most tasks: tau2 only binds DB when `reward_basis` lists it, because
  most hard tasks have no verified gold end-state (`reward.py:_is_read_only`).
- `answer` grades the manager's final text against the task's `answer_spec`; `ask` specs
  need a question naming the ambiguous referent.
- `structural` (DB x ACTION x ENV_ASSERTION, the first reward of this gym) stays in the
  breakdown so earlier runs remain comparable. It passes episodes that answer wrongly or
  write outside the gold chain.

## Tests

```bash
.venv/bin/pytest -q tests/tau2_gym                        # pools, registry, runner
TAU2_DATA_DIR=$PWD/external/tau2_gym/data \
  ~/decomposer_artifacts_new/venvs/tau2/bin/python -m pytest -q tests/tau2_gym/test_tau2_integration.py
```

The integration tests replay the gold actions of a stratified sample of both pools
(`TAU2_FULL_POOL_SCORING=1` for every task, about five minutes) and require 1.0, with an
empty trajectory and a wrong answer both scoring 0.

## Inspect

```bash
OUT=/home/sukhorukov/decomposer_artifacts_new/evaluation/results/tau2_gym/<run>
.venv/bin/python scripts/render_gym_trace.py $OUT/rollouts.jsonl $OUT/traces.md
.venv/bin/python gyms/tau2_gym/analyze_traces.py $OUT/rollouts.jsonl --json $OUT/decomposition.json
```

`analyze_traces.py` reports spawns, waits, and **spawns per wait batch**: subagents
spawned between two consecutive `wait` calls are exactly the ones that ran
concurrently, so that number is the parallelism measure. `share_any_parallel` near 0
means the manager is running a sequential loop with extra steps.

## Why the marker file

`gym_components/pyproject.toml` exists only so that

```python
is_editable_install = (server_dir / "../../pyproject.toml").exists()
```

(`external/Gym/nemo_gym/cli/setup_command.py:126`) is true for an out-of-tree server.
Without it Gym takes the PyPI branch: it installs an unpublished
`nemo-gym==0.5.0rc0` **and** strips every line containing `../..` from
`requirements.txt` (`setup_command.py:154-166`), silently dropping the editable tau2
dependency. Deleting that file breaks component venv creation with a confusing
resolver error.

## The tool-call parser is load-bearing

Qwen3.5 emits tool calls as `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`.
Serving it with `--tool-call-parser hermes` does **not** fail loudly: vLLM returns the
XML as ordinary assistant text, `subagent_runs[].tool_calls` stays empty, the verifier
sees zero predicted calls, and every task scores 0 while the run looks healthy. Use
`qwen3_xml`, which is what every vLLM command in `run.py` pins.

Symptom to watch for: `breakdown.num_predicted_tool_calls == 0` on every rollout.

## Known gaps

- No `run_eval.py` (MLSpace submission); runs are local to Hertz-2.
- Domain `policy.md` reaches only the manager, which must relay the relevant parts in
  each subtask prompt. If traces show subagents failing for want of policy, append it to
  the subagent system prompt in `subagents/graph.py`.
