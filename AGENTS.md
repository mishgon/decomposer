# AGENTS.md

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
- `fork(agent_id) -> agent_id`: creates an agent of the same type with a copy of its conversation history and state. The external environment remains shared.
- `run(agent_id, prompt) -> agent_run_id`: starts a run of an existing agent and immediately returns its run ID. The same agent can run multiple times, retaining its conversation history across runs.
- `wait() -> [...]`: waits for at least one new run to finish and returns all newly available agent responses since the previous `wait` call. Each result includes the run status and any error. Waiting is bounded by a timeout.

This loop enables fully asynchronous orchestration and execution: Decomposer can launch newly unblocked subtasks without waiting for unrelated agent runs to finish. It can also adapt its decomposition as agent results arrive. See `src/decomposer/core.py` for the implementation.

We plan to train the orchestration model in two stages:

1. Off-policy distillation of a carefully prompted LLM into a smaller model.
2. Reinforcement learning that optimizes final task-solving quality, speed, and cost.

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

## Private lmrouter Access

Codex performs setup, shutdown, restart, file transfer, and verification.
Ask only for missing connection details; do not delegate these steps to the user.

### Hosts and Access

The **application host** runs the model clients. The **gateway** reaches lmrouter
and provides `LLM_PROXY_URL` and `LLM_PROXY_MASTER_KEY`. The **Codex host** runs
this session. Reuse hosts and SSH details established in the conversation.
If unspecified, ask for the application host and gateway. “This machine” means
the Codex host.

Test gateway → application host SSH access and, if needed, Codex host → gateway.
Accept an existing SSH alias or hostname, username, port, and authentication details.
Verify unfamiliar host keys through trusted access. If authentication fails,
identify the connection that needs repair. If Codex cannot reach the gateway,
ask the user to run Codex there.

Inspect the gateway alias's effective `RemoteCommand` and `RequestTTY` settings.
Use its configured launcher to activate the environment, for example
`ssh -tt GATEWAY_ALIAS` for a terminal-based `sweethome` launcher.
Confirm the router key is nonempty without displaying it. Use this activated
environment for gateway commands, tmux sessions, and credential transfer.
Run application-host commands over ordinary SSH from the gateway.

### Connection Path

```text
Application / container → private Unix socket → application loopback :18443
                        → SSH reverse tunnel → gateway loopback :18445
                        → MSS-capped gateway relay → router :443
```

Clients use `https://lmrouter.2a2i.org/v1` and verify its TLS certificate.
Both relays forward encrypted bytes and have no API key. The gateway relay sets
`TCP_MAXSEG=1460` before connecting upstream to avoid the reproduced MTU-related
TLS stalls without root access. If the application host reaches lmrouter directly
and has the key, use direct access without these services or a socket variable.

### Start

Inspect existing tmux sessions, processes, listeners, and the Unix socket first.
Reuse services only if they use the current scripts and the MSS-capped path.
Do not start duplicate services or replace a socket with an active listener.
Locate repository checkouts on both hosts. Transfer missing or outdated scripts
from `scripts/lmrouter/` over SSH, preserving other checkout changes.
The gateway needs `router_tunnel.sh` and `gateway_relay.py`; the application host
needs `socket_relay.py` and its Python environment. Both hosts need OpenSSH and
tmux. The Linux gateway also needs Python 3 and Bash with `wait -n -p` support.

Read the upstream destination from the activated gateway environment and check TLS:

```bash
router_host=$(python3 -c 'import os; from urllib.parse import urlsplit; print(urlsplit(os.environ["LLM_PROXY_URL"]).hostname)')
curl --noproxy '*' --connect-timeout 10 --max-time 20 \
  --connect-to "lmrouter.2a2i.org:443:$router_host:443" \
  https://lmrouter.2a2i.org/v1/models
```

HTTP 401 confirms TLS connectivity without credentials. If the URL is missing
or the check fails, request the gateway's working router configuration.
Small curl handshakes can succeed while Python model handshakes stall;
verify a real model call before reporting success.

On the gateway, reuse a dedicated tunnel key or create one if absent:

```bash
install -d -m 700 "$HOME/.ssh"
ssh-keygen -t ed25519 -f "$HOME/.ssh/lmrouter_tunnel" -N '' -C lmrouter-tunnel
```

Using ordinary SSH access, add its public key to the application user's
`~/.ssh/authorized_keys`. Preserve existing keys and avoid duplicate entries.
Apply these restrictions on the same line:

```text
restrict,port-forwarding,permitlisten="127.0.0.1:18443",permitopen="reserved.invalid:1",command="/bin/false" ssh-ed25519 PUBLIC_KEY lmrouter-tunnel
```

Create the application user's SSH directory and authorized-keys file if absent,
with permissions `0700` and `0600`, respectively.

Keep the private key on the gateway. Configure `APPLICATION_ALIAS` there if needed.
From the gateway checkout, start the launcher with the verified destination:

```bash
tmux -L lmrouter new-session -d -s tunnel \
  "bash '$PWD/scripts/lmrouter/router_tunnel.sh' APPLICATION_ALIAS '$HOME/.ssh/lmrouter_tunnel' '$router_host'"
```

The launcher starts the gateway relay on `127.0.0.1:18445`, reconnects SSH,
and stops both processes when terminated.

Transfer `LLM_PROXY_MASTER_KEY` through ordinary SSH to
`~/.local/share/environment/lmrouter.env` on the application host.
Store a shell-safe assignment with directory permissions `0700` and file
permissions `0600`. Never expose the key in output, logs, or command arguments.
If the activated gateway environment lacks the key, ask how to load it.

From the application checkout, start the Unix socket relay and load credentials:

```bash
tmux -L lmrouter new-session -d -s relay \
  "$PWD/.venv/bin/python '$PWD/scripts/lmrouter/socket_relay.py' --socket '$HOME/.local/share/lmrouter-relay/router.sock' --port 18443"

set -a
source "$HOME/.local/share/environment/lmrouter.env"
set +a
export LLM_PROXY_UNIX_SOCKET="$HOME/.local/share/lmrouter-relay/router.sock"
```

Persist credential loading and the socket export in the user's shell startup file.
New client processes must inherit both variables. Toolathlon Gym mounts the socket
directory automatically; other container launchers need that mount, the key,
and the container-side socket path.

Verify the gateway relay listens on loopback `18445`, the application tunnel
listens on loopback `18443`, and the Unix socket has an active listener.
Load the model environment and run this from the application checkout:

```bash
PYTHONPATH=src .venv/bin/python - <<'PYTHON'
from decomposer.models import create_model

print(create_model("qwen_3_8_flash_next_non_thinking").invoke("Say hello.").content)
PYTHON
```

Report success only after receiving a model response. The services normally
survive Codex termination and SSH disconnection; start them again after host reboot.

### Stop

For tunnel-only shutdown, stop the gateway launcher in its activated environment:

```bash
tmux -L lmrouter send-keys -t tunnel C-c
```

For full shutdown, also stop the application Unix socket relay:

```bash
tmux -L lmrouter send-keys -t relay C-c
```

Skip services already stopped. Wait for shutdown and verify the relevant sessions
and processes have exited. Tunnel shutdown must release gateway port `18445` and
application port `18443`; full shutdown must also remove the Unix socket listener.
If a process remains, terminate only the identified lmrouter service process.
Remove a stale socket file only after confirming it has no listener.
Preserve credentials, tunnel keys, SSH configuration, and shell startup settings.

### Restart

Update the affected scripts in the corresponding checkout, transferring the local
versions over SSH when needed. Stop the old gateway launcher using **Stop**,
verify its ports are released, then start it using **Start** with the destination
from the activated gateway's `LLM_PROXY_URL`.

Keep a healthy application Unix socket relay running during a tunnel restart.
Restart it only if its script changed or the relay is unhealthy. If it is already
stopped, start it. Verify the listeners and a real model response as in **Start**.
For a full restart, perform full **Stop**, then **Start**. Codex completes all steps
before reporting success; it does not ask the user to transfer files or restart services.

<!-- BEGIN agent-style v0.4.2 -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Adapter: AGENTS.md cross-agent standard -->
<!-- Target path: <repo root>/AGENTS.md -->
<!-- Load class: single-file; install_mode: append-block -->

# agent-style v0.4.2 — AGENTS.md adapter

agent-style is a literature-backed English technical-prose writing ruleset for AI agents. This adapter is the compact rule payload that AGENTS.md-aware tools (Codex, Jules, Zed, Warp, Gemini CLI, VS Code, Aider via `.aider.conf.yml`, and others) load at session start.

## Self-Verification Handshake

When asked "is agent-style active?" or "what writing rules apply here?", answer: `agent-style v0.4.2 active: 21 rules (RULE-01..12 canonical + RULE-A..I field-observed); full bodies at .agent-style/RULES.md.`

## Load Statement

This adapter is loaded as the root `AGENTS.md` file at the repository root. AGENTS.md-aware tools do not auto-import a second file; the compact directives below are what reach context. Full rule bodies at `.agent-style/RULES.md` are a human-readable reference but are not auto-loaded by AGENTS.md consumers.

## The 21 Rules (Compact Directives)

Canonical rules (from Strunk & White 1959, Orwell 1946, Pinker 2014, Gopen & Swan 1990):

- **RULE-01 Curse of knowledge**: Name your intended reader; do not assume they share your tacit knowledge.
- **RULE-02 Passive voice**: Prefer active voice when the agent is known and worth naming.
- **RULE-03 Concrete language**: Prefer concrete, specific terms over abstract category words like "factors" or "aspects".
- **RULE-04 Needless words**: Cut filler phrases like "in order to", "due to the fact that", "may potentially".
- **RULE-05 Dying metaphors**: Delete clichés like "pushes the boundaries", "paradigm shift", or "state of the art".
- **RULE-06 Plain English**: Prefer "use" over "leverage", "method" over "methodology", "feature" over "functionality".
- **RULE-07 Positive form**: Prefer "trivial" to "not important", "forgot" to "did not remember"; do not stage "X, not Y" antithesis ("not just X, but Y") for emphasis.
- **RULE-08 Claim calibration**: Calibrate verbs to evidence; do not write "proves" when the evidence is "suggests".
- **RULE-09 Parallel structure**: Express coordinate ideas in the same grammatical form.
- **RULE-10 Related words together**: Keep subject close to verb and modifier close to modified; split long parentheticals.
- **RULE-11 Stress position**: Place new or important information at the end of the sentence.
- **RULE-12 Long sentences**: Split sentences over 30 words; vary length across a paragraph.

Field-observed rules (maintainer observation of LLM output, 2022-2026):

- **RULE-A Bullet overuse**: Keep prose in paragraphs when ideas connect; bullets only for genuine lists; avoid forced 3-item triads.
- **RULE-B Dash overuse**: Do not use em or en dashes as casual sentence punctuation; prefer commas, semicolons, colons, parentheses.
- **RULE-C Same-starts**: Do not open two or more consecutive sentences with the same word.
- **RULE-D Transitions**: Do not open sentences with "Additionally", "Furthermore", "Moreover", "In addition".
- **RULE-E Summary closers**: Do not end every paragraph with a sentence that restates its point.
- **RULE-F Term consistency**: Once you define a term or abbreviation, keep using it; do not alternate synonyms.
- **RULE-G Title case**: Use title case for section and subsection headings; articles and short prepositions stay lowercase.
- **RULE-H Citation discipline (critical)**: Support factual claims with verifiable citation or concrete evidence; never fabricate citations.
- **RULE-I Contractions**: Prefer "it is" / "does not" / "cannot" over "it's" / "doesn't" / "can't" in formal technical prose.

## Escape Hatch

*"Break any of these rules sooner than say anything outright barbarous."* — George Orwell, "Politics and the English Language" (1946), Rule 6. Rules are guides to clarity, not ends in themselves.

## Full Rule Bodies (Canonical)

Full directive text, BAD/GOOD example pairs, and rationale per rule: see `.agent-style/RULES.md` in this project, or https://raw.githubusercontent.com/yzhao062/agent-style/v0.4.2/RULES.md for the pinned canonical source.
<!-- END agent-style -->
