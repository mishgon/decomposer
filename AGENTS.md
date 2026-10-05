# AGENTS.md

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

## Coding

### Default workflow

Use this workflow by default:

- Start from the user's request. Ask questions to clarify missing requirements if needed.
- Propose one small changelist (CL) at a time, following [Google's Small CLs guidance](https://google.github.io/eng-practices/review/developer/small-cls.html). Revise the proposal based on the user's feedback.
- Wait for the user to approve the proposal before implementing it. Present the implementation for review. Address feedback and present the revised implementation for review again.
- Proceed to the next CL only after the user approves the implementation. Repeat until all the user's requirements have been fulfilled.

### Code style

Simplicity first. Always write the minimum code that meets current needs. No speculative features, configuration, or abstractions. No unrequested "flexibility" or "configurability". No error handling for cases the surrounding code already rules out. Extract helpers only when multiple places need the same logic and extraction improves clarity.

Make surgical changes. Don't "improve" adjacent code, comments, or formatting. Don't perform unrequested refactoring. Match existing style, even if you'd do it differently. Remove imports, variables, and functions that *your* changes made unused. Mention pre-existing dead code; don't delete it unless asked.

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
