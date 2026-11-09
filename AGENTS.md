# AGENTS.md

## Durable Work

Keep worthwhile code and documentation recoverable from the Git remote. Review changes before committing, and push commits promptly without waiting for a separate request. Commit and push before launching experiments. Do not leave useful work only on the laptop or an experiment host. If a push fails, report it explicitly and retry when connectivity returns. Exclude credentials, caches, generated artifacts, and large datasets from code commits. Do not overwrite unrelated remote changes.

## Network Workarounds

SSH tunnels are banned in the user's organization. Do not propose, create, or use SSH tunnels (including local, remote, dynamic/SOCKS, or Unix-socket forwarding) as connectivity workarounds. This supersedes earlier instructions permitting tunnels with explicit user approval.

Do not forward SSH agents or route traffic through another host without explicit authorization. A request to diagnose or fix connectivity does not authorize these actions. Do not introduce them through launcher transport settings. Use organization-approved networking solutions.

## Repo structure

- `src/decomposer/`: core Decomposer package. This should stay benchmark- and training-agnostic.
- `examples/`: runnable examples of configuring and using Decomposer.
- `gyms/<gym_name>/`: code for running Decomposer or another agent on one or many tasks from a specific gym and saving *the rawest* logs, traces and evaluation results.
- `evals/<gym_name>/`: code for collecting, saving and visualizing evaluation results for Decomposer or another agent on a specific gym. Reuses code from `gyms/<gym_name>`.
- `sft/<gym_name>/`: code for generating traces for SFT on a specific gym. Reuses code from `gyms/<gym_name>`.
- `sft/`: shared SFT code (full dataset, trainer, templates and configs).
- `opd/`: shared on-policy distillation code. Reuses code from `gyms/<gym_name>`.
- `rl/<gym_name>/`: reinforcement learning on a specific gym. Reuses code from `gyms/<gym_name>`.
- `artifacts/data/`: collected trajectories and episode workspaces ignored by git.
- `artifacts/evals/`: evaluation results and aggregate metrics ignored by git.
- `artifacts/training/`: model checkpoints and training logs ignored by git.
- `external/`: third-party repositories, submodules, or vendored code.
- `tests/`: lightweight checks for reusable code and harness utilities.
- `docs/`: design notes, experiment notes, and persistent documentation.

## Coding

### Default workflow

Use this workflow by default:

- Start from the user's request. Ask questions to clarify missing requirements if needed.
- Propose one small changelist (CL) at a time, following [Google's Small CLs guidance](https://google.github.io/eng-practices/review/developer/small-cls.html). Revise the proposal based on the user's feedback.
- Wait for the user to approve the proposal before implementing it. Present the implementation for review. Address feedback and present the revised implementation for review again.
- Proceed to the next CL only after the user approves the implementation. Repeat until all the user's requirements have been fulfilled.

### Code style

The best code is no code at all. This principle applies to custom code we write and maintain; code provided by external libraries does not count.

Use reputable libraries and frameworks (for example, LangGraph) when they solve the current task. Consult their documentation to find existing solutions, using documentation MCP servers where available (for example, LangChain Docs MCP).

When custom code is necessary, write the minimum needed to meet current requirements. No speculative features, configuration, or abstractions. No unrequested flexibility or configurability. No error handling for cases the surrounding code already rules out. Extract helpers only when multiple places need the same logic and extraction improves clarity.

Keep changes limited to the task. Do not improve adjacent code, comments, or formatting, or perform unrequested refactoring. Match existing style. Remove imports, variables, and functions that your changes made unused. Mention pre-existing dead code; do not delete it unless asked.

### Episode Failures and Run Failures

Do not promote an expected episode failure into a fatal experiment failure. A model can exhaust its time, token, or tool budget, produce an invalid answer, or fail a task. These are outcomes the benchmark measures. Preserve the raw trace and failure reason, apply the benchmark's scoring policy, and continue independent episodes. Do not let one expected failure cancel a batch through an uncaught exception in a task group.

Classify failures by cause, not merely by an exception or a missing score. In WideSeek OPD, exhausting the episode deadline counts as zero reward; a missing score caused by a judge failure does not. Infrastructure failures, corrupt data, token alignment failures, and failed worker cleanup must remain explicit and must not silently become model scores. Use bounded retries for known transient infrastructure failures where supported; abort when continuing would invalidate results or leave resources uncontrolled.

Keep baseline, validation, and training adapters consistent with the same gym's failure and scoring semantics. When fixing this class of bug, test both that an expected failed episode lets the batch finish and that a genuine infrastructure failure remains visible. Fix the classification rather than increasing budgets or broadly swallowing exceptions.

## LangChain Docs MCP

Connect both LangChain documentation MCP servers to my coding agent so it can look up current LangChain, LangGraph, and LangSmith docs and API reference.

Servers to add:

* `docs-langchain`: [https://docs.langchain.com/mcp](https://docs.langchain.com/mcp)
* `reference-langchain`: [https://reference.langchain.com/mcp](https://reference.langchain.com/mcp)

Detect which agent or editor I am using (Claude Code, Cursor, Codex CLI, Claude Desktop, Deep Agents Code, VS Code, Antigravity, or another MCP-compatible client). Use the matching setup from [https://docs.langchain.com/use-these-docs.md](https://docs.langchain.com/use-these-docs.md):

* Claude Code: `claude mcp add --transport http` for each server (project scope by default; use `--scope user` only if I ask for global access).
* Codex CLI: `codex mcp add` with each server URL.
* Cursor, Deep Agents Code, VS Code, or Antigravity: merge both entries into the MCP settings JSON using the field names shown on that page for my client.
* Claude Desktop: add both URLs under Settings > Connectors.

Do not invent alternate MCP URLs. After configuring, confirm both servers are listed and reachable.

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
