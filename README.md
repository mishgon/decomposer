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

The minimal example runs Decomposer with Flash Next non-thinking and Qwen3.5-4B
unlooped thinking workers through lmrouter and a local LangGraph server. Both use
the ready model entries in `src/decomposer/models.py`.
From the repository root, install the development environment:

```bash
uv sync
```

With `LLM_PROXY_MASTER_KEY` set, start the subagent server in another terminal:

```bash
scripts/subagents/serve.sh
```

Then run Decomposer from the repository root:

```bash
uv run python examples/minimal/run.py
```

The final answer is printed and the complete message history is saved to
`examples/minimal/messages.md`. See `examples/minimal/README.md` for details.

## Hosted Models and Private lmrouter Access

The ready model instances in [models.py](src/decomposer/models.py) can be used
in any Python code. Their sampling settings stay in the registry. Credentials
and an optional tunnel are configured once per application host, independently
of gyms, collection or training.

Create a private `~/.local/share/environment/lmrouter.env` containing
`LLM_PROXY_MASTER_KEY=YOUR_KEY`, with file permissions `0600`. Load it before
starting Python:

```bash
set -a
source "$HOME/.local/share/environment/lmrouter.env"
set +a
```

If the application host reaches lmrouter directly, no tunnel is needed.
Otherwise, use a gateway that can reach both lmrouter and the application
host's SSH server. This setup works on any such pair of hosts:

These commands assume Unix hosts with SSH and tmux, plus a Python environment
on the application host. The relay uses Unix sockets.

```text
Application / container -> private Unix socket -> application loopback :18443
  -> SSH reverse tunnel -> gateway -> lmrouter.2a2i.org:443
```

Clients keep `https://lmrouter.2a2i.org/v1` and verify its TLS certificate.
The relay forwards encrypted bytes and has no API key.

### Set Up the Tunnel Once

On the gateway, configure an SSH alias in `~/.ssh/config`:

```text
Host application-router
    HostName APPLICATION_HOST
    User YOUR_USER
    Port 22
```

Use the application's actual SSH port. Generate a dedicated key and verify
the application's SSH host key through your existing access:

```bash
ssh-keygen -t ed25519 -f "$HOME/.ssh/lmrouter_tunnel" -N '' -C lmrouter-tunnel
ssh application-router true
```

Add the public key to the application user's `~/.ssh/authorized_keys`, with
these restrictions on the same line:

```text
restrict,port-forwarding,permitlisten="127.0.0.1:18443",permitopen="reserved.invalid:1",command="/bin/false" ssh-ed25519 PUBLIC_KEY lmrouter-tunnel
```

The key permits the specified reverse listener and disables shell access,
PTY, agent/X11 forwarding and usable local forwarding. Keep its private half
on the gateway. From a repository checkout on that gateway:

```bash
tmux -L lmrouter new-session -d -s tunnel \
  "bash '$PWD/scripts/lmrouter/router_tunnel.sh' application-router '$HOME/.ssh/lmrouter_tunnel'"
tmux -L lmrouter attach -t tunnel
```

The launcher reconnects with a delay between 2 and 30 seconds. Its optional
third argument selects the router hostname or a verified IP when gateway DNS
is unavailable. The fourth selects the application-side listener port; update
`permitlisten` above and the relay's `--port` to match.

On the application host, check the TLS route without credentials:

```bash
curl --connect-to lmrouter.2a2i.org:443:127.0.0.1:18443 \
  https://lmrouter.2a2i.org/v1/models
```

HTTP 401 means the unauthenticated request reached the router. HTTP 502 means
inference is not ready. From a checkout on the application host, start the relay:

```bash
tmux -L lmrouter new-session -d -s relay \
  "$PWD/.venv/bin/python '$PWD/scripts/lmrouter/socket_relay.py' --socket '$HOME/.local/share/lmrouter-relay/router.sock' --port 18443"
export LLM_PROXY_UNIX_SOCKET="$HOME/.local/share/lmrouter-relay/router.sock"
```

The socket directory is private (`0700`). Keep only the socket there. If an
unclean exit leaves a stale socket, confirm that no relay owns it before removal.
Reuse existing listeners and relays rather than starting duplicates.

Both tmux sessions survive laptop disconnection. Detach with Ctrl-b d; stop
the selected service with Ctrl-c. Restart the services after host reboot.
Save the credential loading and socket export in your shell startup file for
future terminals. Services and existing tmux sessions must inherit these values
before importing the registry. Without the socket variable, clients connect directly.

### Use the Registry Anywhere

The same connection works with synchronous `invoke()` and asynchronous
`ainvoke()`. No per-gym model setup is required:

```python
from langchain.agents import create_agent
from decomposer.models import MODELS

agent = create_agent(model=MODELS["Qwen/Qwen3.8-Flash-Next-NVFP4"], tools=[])
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
