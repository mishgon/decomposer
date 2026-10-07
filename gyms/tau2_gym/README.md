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
├── subagents/                # LangGraph subagent graphs (Qwen3.5-4B) and their langgraph.json
├── langgraph_server.py       # serves them per run, without LangGraph file persistence
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
| `decomposer_broad_v1` | train | 4,843 / 209 | every `tasks_hard.json` task − held-out (with reserved domains' `_dsh` copies) − follow-up tasks (101) − ask tasks (638) − order gate (213) |

"canon" is `canon_full_{sft,grpo,dpo}.json`; "dead" are tasks with zero passes in
tau2's solo-4B calibration (`progress/gaia2/unified/full/full_report_*.json`); "held-out"
is `HELDOUT_v2`, `heldout_exec_v5`, `cycle_heldout` and `QUARANTINE`, plus every domain
HELDOUT_v2 reserves. v1 kept only `_user_type_for_domain(d) == "scripted"` domains; v2
drops that filter because the Decomposer only ever sends the task's `first_message`
and tau2's own training forces ScriptedUser on every domain.

The broad pool's order gate replays the gold of every filter/superlative task with a
DB check and two or more writes, in listed and in reversed order. It drops the 204
whose final DB differs (IDs are minted from row counts, and tau2's DB term replays the
gold in listed order, so a correct parallel run would fail) and the 9 whose gold does
not replay. It needs the tau2 venv:
`LOGURU_LEVEL=CRITICAL ~/decomposer_artifacts_new/venvs/tau2/bin/python gyms/tau2_gym/task_pools/build_pool.py decomposer_broad_v1`.

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
| `qwen38_flash_teacher_thinking_low` | the same, thinking with `reasoning.effort` low | teacher | train_v2 |
| `qwen38_flash_thinking_low_teacher_qwen35_4b_unlooped` | the same at effort low without the presence penalty; subagents on the unlooped model's own non-thinking sampling, 8192-token cap | teacher | train_v2 |
| `qwen38_flash_non_thinking_teacher_qwen35_4b_unlooped_thinking` | `models.py` presets: `lmrouter/qwen_3_8_flash_next_non_thinking` manager; `lmrouter/qwen_3_5_4b_unlooped_thinking` subagents (type `subagent_thinking`, no output cap) | teacher | broad_v1 |
| `deepseek_v4_flash_teacher` | DeepSeek-v4-flash via OpenRouter (effort max) | teacher | train_v2 |
| `qwen35_4b_base_student` | untuned Qwen3.5-4B, local vLLM | student | eval_v1 |
| `qwen35_4b_sft_mixed_v3_student` | Qwen3.5-4B SFT (v5 student release), local vLLM | student | eval_v1 |
| `opd_rollout` | the OPD round's checkpoint, local vLLM, untruncated sampling, token ids + logprobs | student | train_v2 |

Every experiment exposes one subagent type, `subagent_non_thinking`, with the SFT
releases' canonical description, so teacher traces, SFT data, student evals and OPD
rollouts all see the same `new`/`fork`/`run`/`wait` schema.

Subagents get the domain policy in their system prompt, because they hold the
environment tools and the manager does not. `subagents/graph.py` takes it from the
row's system message (`policy.md`, which the manager also receives at the top of its
first message) and wraps it as tau2's own agent does: `<instructions>` with the
Decomposer agent prompt, then `<policy>`. The subagent type's description says so,
so the manager need not restate the rules in its prompts.

Subagent sampling is Qwen3.5's general non-thinking preset (0.7/0.8/20, presence 1.5,
no length cap) unless the experiment sets `subagent_sampling`. `run.py` passes it to
the LangGraph server as `DECOMPOSER_SUBAGENT_SAMPLING_JSON` (plus
`DECOMPOSER_SUBAGENT_MAX_COMPLETION_TOKENS`), and `run_status.json` records the
values the graph sends. The unlooped teacher experiment uses the unlooped model's
recommended non-thinking values, 0.7/0.8/20 with no penalties, but caps completions
at 8192 tokens rather than its 2048: in the Workplace smoke 2048 cut legitimate
full-record reports (`QWEN35_UNLOOPED_NON_THINKING`). A subagent completion that reaches the cap fails
that subagent run (`ChatVLLM` raises on `finish_reason=length`).

Qwen3.8-Flash-Next on the proxy's Responses API renders replayed reasoning items: a
probe on 2026-09-25 raised the follow-up input from 83 to 158 tokens with the earlier
reasoning item. Gym's decomposer agent replays every output item, so the thinking
teacher sees its own earlier reasoning; experiments that rely on it set
`upstream_replays_reasoning`. (Qwen3.6 on the same proxy discards them; see the
Workplace README.)

Before starting, `run.py` checks that the shared proxy lists every model the run
requests (`GET /models`), because the local proxies retry upstream failures and a
missing model would otherwise stall the run.

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

# a student checkpoint on the held-out pool, five domain-stratified tasks per domain,
# evaluated in one command: the run, then metrics into <run>/eval_metrics.json
.venv/bin/python -m evals.tau2_gym.run --experiment qwen35_4b_sft_mixed_v3_student \
  --tasks-per-domain 5 --num-repeats 3 --manager-gpu 6

# OPD rollouts from a round's checkpoint (what opd/ drives)
.venv/bin/python gyms/tau2_gym/run.py --experiment opd_rollout --manager-checkpoint <dir> \
  --manager-gpu 5 --output-dir <round>/rollouts
```

`--pool` overrides the experiment's pool and `--tasks-per-domain k` takes a
deterministic, hash-stratified subsample. The run name records experiment, pool,
subsample and repeats. `run_status.json` records the experiment, pool sha, generated
config sha and, for local managers, a fingerprint of the served checkpoint.

## Subagent backends

An experiment with a `models.py` preset (`manager_preset`, `subagent_preset`; see
`gyms/tau2_gym/model_presets.py`) takes that role's sampling from the preset alone. The manager
preset becomes the manager proxy's extra body (non-thinking becomes `reasoning.effort:
"none"`, since the proxy ignores `chat_template_kwargs` on the Responses API). A
subagent preset runs as the `preset` backend: its graph is `create_model(preset)` itself,
which reaches the shared proxy on the preset's endpoint with `LLM_PROXY_MASTER_KEY`, so
no local subagent proxy starts.

Otherwise `--subagent-backend` picks where subagent traffic goes. The subagent graph is identical
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

## The subagent server keeps no state on disk

`langgraph dev` saves its in-memory store to `.langgraph_api/` in its working directory,
reloads it at startup and rewrites all of it every 10 s. Left in `subagents/` across
runs, that store reached 3.8 GB, and `threads.get_history` (which `wait` calls for every
finished run) took 7-35 s: subagents finished in ~15 s but reached the manager after
145-215 s. `langgraph dev` cannot turn this off (the CLI drops `disable_persistence`
from `langgraph.json`), so `run.py` starts `langgraph_server.py`, which calls
`run_server(disable_persistence=True)` from `<run>/langgraph/`, as the GAIA2 gym does.
The graphs still come from `subagents/langgraph.json`.

## How scoring works

Every subagent tool call reaches the resources server with the rollout's session
cookie, and the server logs each call it executes with its result, in execution
order. `verify` scores that log, not the calls the Decomposer reports. This is how
tau2-gym's own environment records its trajectory, and how the GAIA2 judge reads
ARE's event log: the environment is the source of truth.

The reported calls are the wrong source. The Decomposer rebuilds them from subagent
histories (`decomposer_agent/app.py` `_collect_subagent_tool_calls`), and a subagent
that dies usually leaves no history. On the Qwen3.8 teacher run of 2026-09-23
(1,239 rollouts), subagents executed 34,485 calls but only 26,704 were reported;
55 of 92 failed subagent runs reported none, and 35 of the 53 rollouts with a failed
subagent had scored 1 without their writes being checked.

`tau2_bridge.score_logged_calls` turns the log into the trajectory of a *virtual flat
agent*: each call with its recorded result, then the manager's final report as the
agent's final message. That is exactly what tau2's own `training/reward.compute_reward`
consumes, so the Decomposer is graded by tau2's binary predicate with its tri-state
terms (`None` = does not apply to this task). The log must also reproduce: it is
replayed into a freshly seeded environment, and any call whose result differs scores
the rollout 0 (`log_replay_error`), as tau2's DB term fails closed.

Every breakdown carries `scoring`: `server_log_v1` for this path, or
`reported_replay_v0` for the earlier replay of reported calls. Runs from before the
change have no log and cannot be rescored. `logged_calls`, `reported_calls` and
`unreported_calls` show how much the Decomposer's own view missed;
`python -m evals.tau2_gym.run --metrics-only <run>` summarises them per run.

The terms:

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
.venv/bin/python -m evals.tau2_gym.run --metrics-only $OUT        # pass@1/@k/^k overall and per domain
.venv/bin/python -m evals.tau2_gym.analyze_traces $OUT/rollouts.jsonl --json $OUT/decomposition.json
```

`evals/tau2_gym/analyze_traces.py` reports spawns, waits, and **spawns per wait batch**: subagents
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
