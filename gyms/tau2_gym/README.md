# tau2 gym

Runs the Decomposer on Timur's tau2 GAIA2-hard domains, through NeMo-Gym.

The environment is exposed as a **NeMo-Gym resources server hosted outside the
`external/Gym` submodule**. `gym env start` finds it because `NEMO_GYM_EXTRA_ROOTS`
points at `gym_components/`, and extra roots are searched before Gym's own tree
(`nemo_gym/__init__.py:52-79`). No submodule edits are required.

```
gyms/tau2_gym/
├── run.py                    # local runner: vLLM -> langgraph -> gym env -> gym eval -> validate
├── tau2_export.py            # builds the Gym dataset (runs inside the tau2 venv)
├── analyze_traces.py         # decomposition / parallelism statistics
├── configs/                  # Hydra config, one per experiment
├── subagents/                # LangGraph subagent server (Qwen3.5-4B, non-thinking)
└── gym_components/           # <- NEMO_GYM_EXTRA_ROOTS
    ├── pyproject.toml        # marker; see "Why the marker file" below
    └── resources_servers/tau2_gym/
        ├── app.py            # seed_session / POST {tool} / verify
        ├── tau2_bridge.py    # all tau2 imports: tasks, envs, schemas, scoring
        └── requirements.txt
```

## One-time setup

The submodule is already wired (`external/tau2_gym`, from a local bundle, mirroring
`external/gaia2`). Hertz-2 has no GitLab access, so refresh it by regenerating the
bundle:

```bash
git -C /home/sukhorukov/tau2-gym bundle create /home/sukhorukov/tau2-gym.bundle --all
git -C external/tau2_gym fetch origin
```

Create the tau2 venv used by `tau2_export.py` (tau2's dependency set is heavy, so it
is deliberately kept out of the project venv):

```bash
source ~/proxy_on.sh
uv venv --python 3.12 /home/sukhorukov/decomposer_artifacts_new/venvs/tau2
VIRTUAL_ENV=/home/sukhorukov/decomposer_artifacts_new/venvs/tau2 \
  uv pip install -e external/tau2_gym
```

## Run

`run.py` needs `OPENROUTER_API_KEY_DECOMPOSER` exported for the DeepSeek manager.

```bash
# what would run, without running it
.venv/bin/python gyms/tau2_gym/run.py --dry

# one task end to end
export OPENROUTER_API_KEY_DECOMPOSER=...
.venv/bin/python gyms/tau2_gym/run.py --limit 1 --cuda-visible-devices 7

# the milestone-1 sweep: 3 domains x 5 tasks, 3 repeats
.venv/bin/python gyms/tau2_gym/run.py --num-repeats 3 --cuda-visible-devices 7
```

Defaults are three GAIA2-hard domains — `hotel_reservations`, `library_lending`,
`gym_memberships` — chosen because they are in `_EXPLICIT_AUTO_DOMAINS`
(`external/tau2_gym/training/tau2_env_manager.py:60-68`), so they run with
`ScriptedUser`: a single compound user turn and **no user-simulator LLM**. Domains
outside that set need a user model and are not single-turn.

## Subagent backends

`--subagent-backend` picks where subagent traffic goes. The subagent graph is identical
either way: it always talks plaintext HTTP to loopback with `api_key="EMPTY"`, resolving
its endpoint from `TAU2_GYM_MODEL_BASE_URLS_JSON`.

| | `local_vllm` (default) | `llm_proxy` |
|---|---|---|
| Model host | dedicated vLLM this runner starts | shared Qwen3.5-4B replicas |
| GPU | one, held for the whole run | **none** |
| `model_startup` | 140-360 s | **~2 s** |
| Needs | a free GPU + port | `LLM_PROXY_URL`, `LLM_PROXY_MASTER_KEY` |

```bash
source ~/.secrets/decomposer.env      # LLM_PROXY_URL + LLM_PROXY_MASTER_KEY
.venv/bin/python gyms/tau2_gym/run.py --subagent-backend llm_proxy --num-repeats 3
```

`llm_proxy` starts `gyms.remote_model_proxy` on port 8143 (8142 is the workplace gym's
manager proxy). That local process injects the credentials and terminates the shared
proxy's **self-signed** TLS, so the master key never enters the LangGraph process. It also
retries `{408,429,5xx}` with backoff, which matters because the replicas are shared.

Two proxy flags are deliberately **not** passed:

- `--response-tool-parser` — its Qwen XML normalisation only runs on the Responses API
  path (`remote_model_proxy.py:317`). Subagents use Chat Completions, and the upstream
  vLLM already applies `qwen3_xml`, so its responses are structured; normalising again
  would be wrong.
- `--extra-body-json` — it merges server-side and *overrides caller keys*
  (`remote_model_proxy.py:49-54`). Leaving it empty preserves the graph's own sampling so
  the two backends stay comparable.

Measured on 3 domains x 5 tasks x 3 repeats, the backends agree within noise:

| | local_vllm | llm_proxy |
|---|---|---|
| pass rate | 0.756 | 0.800 |
| spawns/rollout | 1.58 | 1.64 |
| max fan-out (mean / max) | 1.33 / 4 | 1.20 / 4 |

Per-rollout latency is roughly 2x higher through the proxy (shared replicas, remote host),
but the ~6 minute model startup disappears.

## Inspect

```bash
OUT=/home/sukhorukov/decomposer_artifacts_new/evaluation/results/tau2_gym/<run>
.venv/bin/python scripts/render_gym_trace.py $OUT/rollouts.jsonl $OUT/traces.md
.venv/bin/python gyms/tau2_gym/analyze_traces.py $OUT/rollouts.jsonl --json $OUT/decomposition.json
```

`render_gym_trace.py` is already Decomposer-aware. `analyze_traces.py` is the new
part: it reports spawns, waits, and **spawns per wait batch** — subagents spawned
between two consecutive `wait` calls are exactly the ones that ran concurrently, so
that number is the parallelism measure. `share_any_parallel` near 0 means the
manager is running a sequential loop with extra steps.

## How scoring works

`verify` receives a *flattened* trajectory: every subagent `function_call` in report
order plus the final assistant message, and **no `function_call_output` items**
(`decomposer_agent/app.py:287-311`).

That rules out tau2's usual path. `Environment.set_state` requires a `ToolMessage`
after every tool call and raises when a replayed output differs from the recorded
one (`environment.py:362-410`), so `compute_reward_from_evaluators` cannot consume
the flattened view. Instead `tau2_bridge.score_trajectory` rebuilds the predicted
database by **replaying the predicted calls into a fresh environment**, symmetrically
with the gold replay — the same shape as `workplace_assistant`'s `is_correct`.

Reward is `DB x ACTION x ENV_ASSERTION`, each binarised. The FORMAT / TOOL_VALIDITY /
SIGNAL terms of `reward.py:1093` are deliberately omitted: they describe a single flat
agent's own text output, so under the Decomposer they would measure *subagent*
formatting rather than the manager.

Sanity check (a gold replay must score 1.0):

```bash
TAU2_DATA_DIR=$PWD/external/tau2_gym/data \
PYTHONPATH=$PWD/gyms/tau2_gym/gym_components/resources_servers/tau2_gym \
/home/sukhorukov/decomposer_artifacts_new/venvs/tau2/bin/python -c "
import tau2_bridge as B
from tau2.data_model.message import ToolCall
t = B.load_task('gym_memberships', 'hw0_filt_0')
calls = [ToolCall(id=str(i), name=a.name, arguments=a.arguments, requestor=a.requestor)
         for i, a in enumerate(t.evaluation_criteria.actions)]
print(B.score_trajectory('gym_memberships', 'hw0_filt_0', calls))
print(B.score_trajectory('gym_memberships', 'hw0_filt_0', []))
"
```

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
`qwen3_xml`, which is what `gyms/workplace_assistant/experiments.py` pins for the same
model. `gyms/remote_model_proxy.py:170-196` exists to normalise this same format.

Symptom to watch for: `breakdown.num_predicted_tool_calls == 0` on every rollout.

## Known gaps

- `data/sft/adapters/tau2.py` does not exist yet, so these traces cannot be built
  into an SFT release. That is the immediate next step.
- No `run_eval.py` (MLSpace submission) and no `tests/tau2_gym/`.
- Domain `policy.md` currently reaches only the manager, which must relay the
  relevant parts in each subtask prompt. If traces show subagents failing for want of
  policy, append it to the subagent system prompt in `subagents/graph.py`.
