# AGENTS.md

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
- `fork(subagent_id) -> subagent_id`: creates a subagent of the same type with a copy of its conversation history and state. The external environment remains shared.
- `run(subagent_id, prompt) -> subagent_run_id`: starts a run of an existing subagent and immediately returns its run ID. The same subagent can run multiple times, retaining its conversation history across runs.
- `wait() -> [...]`: waits for at least one new run to finish and returns all newly available subagent responses since the previous `wait` call. Each result includes the run status and any error. Waiting is bounded by a timeout.

This loop enables fully asynchronous orchestration and execution: Decomposer can launch newly unblocked subtasks without waiting for unrelated subagent runs to finish. It can also adapt its decomposition as subagent results arrive. See `src/decomposer/core.py` for the implementation.

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

The **application host** is the machine where lmrouter access is needed.
The **gateway** already reaches lmrouter and has `LLM_PROXY_URL` and
`LLM_PROXY_MASTER_KEY` in its environment. The **Codex host** runs the
current session.

When asked to configure lmrouter, ask the user to specify the application
host and gateway unless they have already done so. Use the gateway specified
by the user. If the request says "this machine", use the Codex host as the
application host.

Ask only for missing SSH details for gateway → application host and,
if Codex runs elsewhere, Codex host → gateway. Accept an existing alias
on the connecting host or a hostname, username, and port, together with
any required authentication details. Test these supplied connections.
If authentication fails, explain which connection the user needs to
configure. If Codex host → gateway access is impossible, ask the user
to run Codex on the gateway.

Activate the gateway's configured user environment. Inspect the supplied alias's
effective SSH settings, including `RemoteCommand` and `RequestTTY`, before
running remote commands. A launcher such as `sweethome` may activate the
environment that provides `LLM_PROXY_MASTER_KEY` and changes the user's home.
Connect using the configured launcher. If activation requires a terminal,
use `ssh -tt GATEWAY_ALIAS` and run commands inside the activated shell.
Confirm that `LLM_PROXY_MASTER_KEY` is nonempty without displaying its value.
Use this activated environment for setup and credential transfer.

The user supplies host identities and ordinary SSH access. Perform the
remaining setup yourself: prepare the scripts and runtime, configure the
restricted tunnel key, tunnel, relay, credentials, and persistent environment,
then verify a model call. Check the supplied hosts and connections and ask
the user for any missing access information.

Use a reverse tunnel and a private Unix socket relay:

```text
Application / container → private Unix socket → application loopback :18443
                        → SSH reverse tunnel → gateway → lmrouter.2a2i.org:443
```

Clients keep `https://lmrouter.2a2i.org/v1` and verify its TLS certificate.
The relay forwards encrypted bytes and has no API key. Reuse working
listeners and relays. If the application host already reaches lmrouter
and has the key, use direct access without a tunnel or socket variable.

Before setting up the tunnel, read `LLM_PROXY_URL` in the activated gateway
environment. Use its hostname or IP as the tunnel destination.
Keep the model registry URL unchanged.
Verify the destination with TLS for `lmrouter.2a2i.org` before proceeding:

```bash
router_host=$(python3 -c 'import os; from urllib.parse import urlsplit; print(urlsplit(os.environ["LLM_PROXY_URL"]).hostname)')
curl --noproxy '*' --connect-timeout 10 --max-time 20 \
  --connect-to "lmrouter.2a2i.org:443:$router_host:443" \
  https://lmrouter.2a2i.org/v1/models
```

HTTP 401 confirms connectivity and successful TLS verification without a key.
Keep certificate verification enabled. If `LLM_PROXY_URL` is missing or the
check fails, ask the user for the gateway's working router configuration.

Run commands from repository checkouts on the corresponding hosts. Locate
existing checkouts or copy the two tunnel scripts if needed. Execute
application-host commands over ordinary SSH from the gateway.

On the gateway, reuse an existing dedicated tunnel key or create one:

```bash
ssh-keygen -t ed25519 -f "$HOME/.ssh/lmrouter_tunnel" -N '' -C lmrouter-tunnel
```

Using ordinary SSH access, add its public key to the application user's
`~/.ssh/authorized_keys` with these restrictions on the same line. Preserve
existing keys and avoid duplicate entries:

```text
restrict,port-forwarding,permitlisten="127.0.0.1:18443",permitopen="reserved.invalid:1",command="/bin/false" ssh-ed25519 PUBLIC_KEY lmrouter-tunnel
```

Keep the private key on the gateway. Configure an SSH alias there for the
application host using the supplied hostname, user, and port if no suitable
alias exists. Verify unfamiliar SSH host keys through trusted access.
Start the tunnel with that alias, the dedicated key, and the destination
derived from `LLM_PROXY_URL`:

```bash
tmux -L lmrouter new-session -d -s tunnel \
  "bash '$PWD/scripts/lmrouter/router_tunnel.sh' APPLICATION_ALIAS '$HOME/.ssh/lmrouter_tunnel' '$router_host'"
```

The launcher reconnects automatically. Use ordinary SSH access, rather than
the restricted tunnel key, to transfer `LLM_PROXY_MASTER_KEY` from the gateway
environment to `~/.local/share/environment/lmrouter.env` on the application
host. Store a shell-safe assignment, with directory permissions `0700` and
file permissions `0600`. Never expose the key in messages, logs, or command
arguments. If the variable is unavailable after activating the configured
environment, ask the user how to load it.

On the application host, check the TLS route without credentials:

```bash
curl --noproxy '*' --connect-timeout 10 --max-time 20 \
  --connect-to lmrouter.2a2i.org:443:127.0.0.1:18443 \
  https://lmrouter.2a2i.org/v1/models
```

HTTP 401 confirms that the request reached the router. Start the relay and
load the model environment:

```bash
tmux -L lmrouter new-session -d -s relay \
  "$PWD/.venv/bin/python '$PWD/scripts/lmrouter/socket_relay.py' --socket '$HOME/.local/share/lmrouter-relay/router.sock' --port 18443"

set -a
source "$HOME/.local/share/environment/lmrouter.env"
set +a
export LLM_PROXY_UNIX_SOCKET="$HOME/.local/share/lmrouter-relay/router.sock"
```

Persist credential loading and the socket export in the user's shell startup
file for future sessions. New processes must inherit both variables.
Toolathlon Gym mounts the socket directory automatically; other container
launchers need the directory mount, key, and container-side socket path.

Verify a real model call on the application host:

```bash
PYTHONPATH=src .venv/bin/python - <<'PYTHON'
from decomposer.models import create_model

model = create_model("qwen_3_8_flash_next_non_thinking")
print(model.invoke("Say hello.").content)
PYTHON
```

Report success only after receiving a model response. Both tmux sessions
normally survive Codex termination and SSH disconnection. Restart them
after host reboot.

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
