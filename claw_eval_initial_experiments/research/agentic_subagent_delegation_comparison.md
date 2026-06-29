# Subagent and Task Delegation Architectures

Date: 2026-06-25

This report compares how ZeroClaw, opencode, and OpenHands implement subagents
and task delegation. The focus is operational behavior: what child agents see,
what tools they can use, how results flow back to the parent, how async work is
handled, and which ideas are worth transferring to the decomposer.

## Source Basis

- ZeroClaw: local Rust source snapshots inspected under `/tmp/zeroclaw_*.rs`,
  especially `spawn_subagent.rs`, `delegate.rs`, `loop.rs`, and history/tool
  result collection helpers.
- opencode: `anomalyco/opencode` `dev` source snapshots, especially
  `packages/opencode/src/tool/task.ts`, `task.txt`,
  `agent/subagent-permissions.ts`, `session/message-v2.ts`, and
  `session/compaction.ts`.
- OpenHands: `openhands-sdk==1.29.0` and `openhands-tools==1.29.0` PyPI source
  distributions, especially `openhands/tools/delegate/*`,
  `openhands/tools/task/*`, `openhands/sdk/subagent/*`,
  `openhands/sdk/agent/*`, and `openhands/sdk/context/condenser/*`.
- OpenHands top-level repository currently points agent/server source to
  `OpenHands/software-agent-sdk`: https://github.com/OpenHands/OpenHands and
  https://github.com/OpenHands/software-agent-sdk.

## Executive Summary

The three systems converge on the same important invariant: subagent history
stays separate from parent history. The parent receives a bounded result object
or final message, not the child agent's full transcript.

| System | Main primitive | Child context | Result returned to parent | Async behavior |
| --- | --- | --- | --- | --- |
| ZeroClaw | `spawn_subagent` and `delegate` | `spawn_subagent` gets a fresh prompt with inherited identity/policy; `delegate` gets a constructed prompt for a named target agent | Final child output as tool result; background `delegate` result stored as JSON | `spawn_subagent` is synchronous; `delegate` supports sync, background polling, and parallel fan-out |
| opencode | `task` | Fresh child session unless `task_id` resumes prior child session | XML-like `<task>` tool result containing final child text or error | Foreground waits; experimental background mode returns running result and later auto-injects completion |
| OpenHands | `delegate` and `task` | `delegate` uses persistent named child conversations; `task` creates/resumes task conversations | `DelegateObservation` or `TaskObservation` containing final child response/error | `delegate` runs multiple child tasks in parallel threads and waits; `task` is blocking |

Design implication for the decomposer: do not append raw child transcripts to
the parent. Treat subagents as isolated workers that return a compact,
well-specified artifact.

## ZeroClaw

### Primitives

ZeroClaw has two separate primitives:

- `spawn_subagent`: an ephemeral child run under the parent's identity and
  permissions.
- `delegate`: a named-agent delegation tool that can use different configured
  agents, models, runtime profiles, and execution modes.

They are not equivalent. `spawn_subagent` is a lightweight "same agent, fresh
context" fork. `delegate` is the orchestration surface for specialized agents,
background work, result polling, cancellation, and parallel fan-out.

### `spawn_subagent`

`spawn_subagent` exposes a single required argument: `prompt`. The tool
description explicitly says the prompt must be self-contained because the
subagent does not see the parent conversation history.

Execution shape:

- It refuses if called from a subagent, using an `is_subagent_caller` depth-1
  guard.
- It enforces the caller's risk profile: if `spawn_subagent` is excluded or not
  allowlisted when an allowlist exists, the spawn is rejected before launch.
- It consumes an action budget slot through the shared security policy.
- It builds a `SubAgentSpawn` context from the parent alias and live
  `SecurityPolicy`.
- It calls the normal agent loop with:
  - the same parent alias
  - `Some(prompt)` as the initial task
  - a synthetic session path like `subagent-{uuid}`
  - `AgentRunOverrides { security: Some(...), memory: None, is_subagent: true }`

The child differs from the parent mainly by context and run flags, not by
identity. It inherits the parent's identity, security policy, workspace
boundary, and memory allowlist. The child is marked as a subagent so its own
`spawn_subagent` tool refuses recursive spawns.

Tool access is therefore mostly the parent's normal tool registry after policy
filtering. The important exception is recursive `spawn_subagent`, which is
blocked. If other reentrant tools are present, their own policies decide whether
they can be used.

Result shape:

- The parent awaits the child run.
- On success, `ToolResult.output` is the final response string, or the fallback
  text `subagent completed without output`.
- On failure, `ToolResult.error` contains `subagent run failed: ...`.
- The parent does not receive the child message history.

This answers the earlier concern directly: even if the parent already had raw
logs in its context, `spawn_subagent` does not magically remove those logs. It
adds only the child final answer as a tool result. The value is not compression
of already-present parent context; the value is isolated reasoning over a
self-contained prompt and a smaller answer surface for later turns.

### `delegate`

`delegate` is richer and more explicit. Its schema supports these actions:

- `delegate`: run work now.
- `check_result`: retrieve a background task result.
- `list_results`: list background task records.
- `cancel_task`: cancel a running background task.

It also supports execution modifiers:

- `background: true`: detach the run and return a `task_id`.
- `parallel: [...]`: run the same prompt across multiple named agents
  concurrently.
- `agentic`: resolved from the target agent runtime profile; when true, the
  target runs a tool-call loop rather than a single model call.

Synchronous non-agentic delegation is a single model call. ZeroClaw builds an
enriched system prompt for the target agent, optionally prepends a `[Context]`
section to the prompt, calls the target model, and returns:

```text
[Agent '<agent>' (<provider>/<model>)]
<response>
```

Agentic delegation is a child tool loop. It builds a local history consisting
of the enriched system prompt and the delegated user prompt. It filters the
parent tool registry through the target risk profile, excludes `delegate`
itself, and runs `run_tool_call_loop` silently. The parent receives only the
final response string, prefixed with target metadata.

Tool access for `delegate` is therefore stricter and more specialized than
`spawn_subagent`:

- non-agentic delegates have no tool loop
- agentic delegates get a filtered subset of parent tools
- the `delegate` tool is removed from the agentic child registry
- memory tools can be re-bound to target-specific memory namespaces
- the target policy is checked to prevent delegation from widening privilege

### Async and Background

`spawn_subagent` is synchronous. The parent awaits the run.

`delegate` has three execution modes:

- Single-agent sync: run one named target and return its result.
- Parallel: spawn one async task per selected named agent, then await all task
  handles and return a combined report. The parent does not choose winners or
  exclude slower agents; it waits for every launched child to finish or fail.
- Background: create a UUID task id, write an initial `Running` JSON record
  under `workspace/delegate_results/{task_id}.json`, spawn a tokio task, and
  return immediately.

Background mode means the parent/user/session must later retrieve or cancel the
task. The returned instruction is explicit: call `check_result` with the
`task_id`. There is supervision and cancellation support, but there is no
opencode-style automatic injection of the completed background result into the
parent conversation.

### Context and History

The parent history receives the normal tool-call round: assistant tool call,
then tool result. ZeroClaw collects tool results, truncates long outputs using
`max_tool_result_chars`, and appends the tool output to the parent loop history.

The child run's own history is local to that run. For `spawn_subagent`, the
parent receives the child's final response only. For `delegate`, the parent
receives a final response or serialized background result, not the child
transcript.

Context management is mostly trimming, not semantic compaction. The tool loop
keeps a `Vec<ChatMessage>` history and can trim to recent turns against a
context token budget. Tool outputs are separately bounded by max result
characters. There is no first-class child-summary compactor comparable to
opencode or OpenHands.

### Best Read of ZeroClaw's Design

Use `spawn_subagent` when the same agent needs a clean scratchpad for a focused
self-contained prompt. Use `delegate` when the parent needs named specialists,
different models, background execution, cancellation, or parallel fan-out.

The split is useful only if those semantics stay distinct. If `spawn_subagent`
starts accumulating routing, background, and role-specific permissions, it
becomes redundant with `delegate`.

## opencode

### Primitive

opencode uses one main subagent primitive: the `task` tool. It does not have a
separate `spawn_subagent` and `delegate` distinction.

The task tool arguments include:

- `description`: short label
- `prompt`: task for the subagent
- `subagent_type`: required type of agent
- `task_id`: optional resume target
- `background`: optional and only available behind
  `OPENCODE_EXPERIMENTAL_BACKGROUND_SUBAGENTS=true`

The bundled task instructions tell the parent to use the tool for complex,
multi-step autonomous work, not for simple file reads or searches. They also
tell the parent to launch multiple agents concurrently where possible by
issuing multiple tool uses in a single model message.

### Child Session Model

A task creates or resumes a child session:

- New task: create a session with `parentID` set to the parent session.
- Resume: load the session whose id is provided as `task_id`.
- Child title includes the description and subagent name.
- Child agent is the selected `subagent_type`.
- Child model is either the subagent's configured model or the parent's current
  model/provider.

The child does not automatically receive the parent's full transcript. Fresh
task invocations start with fresh context. Resume continues the previous child
session and therefore includes that child session's previous messages and tool
outputs.

The prompt text is the delegation contract. opencode's own instructions tell
the parent to specify exactly what the child should return in its final message.

### Tools and Permissions

opencode derives child session permissions from both parent restrictions and
subagent configuration:

- parent deny rules are propagated
- parent external-directory restrictions are propagated
- the subagent's own permissions determine its positive capabilities
- `todowrite` is denied by default unless the subagent explicitly allows it
- `task` is denied by default unless the subagent explicitly allows it
- experimental primary tools can be denied for subagents

This means recursive task spawning is disabled by default for subagents. It can
be enabled intentionally by subagent permission configuration.

### Result Shape

The task output is rendered as an XML-like block:

```xml
<task id="..." state="completed">
<task_result>
...
</task_result>
</task>
```

For running or failed work, the same envelope uses `state="running"` or
`state="error"` and a `task_error` tag for errors.

The parent receives this rendered tool result. The user does not automatically
see the child output. The parent is expected to summarize or use the result in
its own final answer.

### Async and Background

Foreground task execution uses a background job internally but the tool call
waits for completion or promotion. From the parent's perspective, foreground
task is synchronous.

Experimental background mode is different:

- `background: true` requires the experimental environment flag.
- The tool immediately returns a running `<task>` result.
- A background waiter later injects a synthetic completed/error task result
  into the parent session.
- The parent is explicitly told not to sleep, poll, or duplicate the task's
  work.

This is a stronger UX than ZeroClaw background delegation. Responsibility is
not pushed onto the parent to remember a polling call; completion is delivered
back into the conversation automatically.

### Context and History

opencode stores rich session parts rather than just flat text:

- user text and attachments
- assistant text
- reasoning parts
- tool calls and tool results
- subtask/task parts
- compaction parts

When converting stored messages back to model input, opencode can include
assistant reasoning when appropriate for the same model, preserve tool call
structure, mark interrupted pending tool calls, and clear old compacted tool
outputs.

Its compaction system summarizes older context while preserving a recent tail.
The compaction flow can prune old tool outputs and uses a dedicated compaction
path rather than relying only on hard truncation.

### Best Read of opencode's Design

opencode treats subagents as resumable child sessions invoked through a single
task abstraction. This keeps the parent mental model simple:

- use `task` for substantial delegated work
- use `task_id` to continue the same child
- use background only for independent work
- expect one final answer from the child

The main strength is the simple public interface combined with separate child
history and automatic background result injection.

## OpenHands

### Two Delegation Surfaces

OpenHands has two subagent surfaces:

- `delegate`: lower-level, explicit `spawn` then `delegate`.
- `task`: higher-level, blocking task tool backed by `TaskManager`.

They overlap but serve different orchestration styles.

`delegate` is for long-lived named child conversations. `task` is for
one-shot or resumable task execution.

### `delegate`: Explicit Spawn Then Delegate

The delegate action schema has:

- `command`: `spawn` or `delegate`
- `ids`: required for `spawn`
- `agent_types`: optional types corresponding to the ids
- `tasks`: required for `delegate`, mapping child ids to task prompts

The `DelegateExecutor` owns:

- a parent `LocalConversation`
- a map of child id to `LocalConversation`
- a default child limit of 5
- an optional confirmation handler for user approvals

`spawn` creates child conversations but does not run tasks yet:

- resolves each requested agent type through the subagent registry
- copies the parent LLM, resets metrics, and disables streaming
- creates the worker agent from the selected factory
- gives the child the same workspace path as the parent
- inherits persistence under a `subagents` directory when the parent persists
- applies the agent definition's confirmation policy, or inherits the parent
  policy if none is specified
- stores the child conversation under the caller-provided id

`delegate` then sends tasks to existing child ids:

- validates that all requested ids were spawned
- creates one Python thread per delegated task
- sends each task into that child conversation
- runs each child conversation until finished, handling confirmation requests
- joins all threads
- extracts each child's final response from its event history
- returns a consolidated `DelegateObservation`

This is parallel fan-out with blocking collection. The parent waits until every
thread finishes. There is no winner selection or cancellation of slower
children in the normal path.

Because child conversations remain in the executor map, repeated delegation to
the same id continues that child conversation's history.

### `task`: TaskManager-Based Subagent Runs

The `task` tool action has:

- `description`
- `prompt`
- `subagent_type`, defaulting to `general-purpose`
- `resume`, a prior task id
- deprecated `max_turns`, ignored

`TaskManager.start_task` is blocking:

- attaches to the parent conversation on first use
- creates a new task or resumes a saved one
- sends the prompt to the child conversation
- runs until finished, including confirmation handling
- extracts the final response or error
- syncs child metrics into parent conversation stats
- evicts/closes the live child conversation while preserving task metadata

Fresh tasks get ids like `task_00000001` plus a UUID conversation id. If the
parent has persistence, child state lives under the parent's `subagents`
directory. Otherwise OpenHands uses a temporary directory. Resuming a task
reopens a `LocalConversation` with the saved conversation id and persistence
directory.

The parent receives a `TaskObservation` containing:

- task id
- subagent type
- status
- final result or error text

The parent does not receive the child transcript.

### Subagent Definitions, Tools, and Permissions

OpenHands subagents are registered from code, plugins, or Markdown definition
files. File-based agents can live under project or user `.agents/agents` and
`.openhands/agents` directories. Project definitions take priority.

Markdown frontmatter supports:

- `name`
- `description`
- `model`
- `tools`
- `skills`
- `max_iteration_per_run`
- `max_budget_per_run`
- `hooks`
- `profile_store_dir`
- `mcp_servers`
- `permission_mode`
- `condenser`

The registry converts a definition into an agent factory:

- `model: inherit` keeps the parent LLM; an explicit model loads an LLM profile
- listed tool names are resolved into actual tools
- listed skills are resolved from project/user skill directories
- MCP servers are attached to the agent config
- the definition's system prompt becomes an agent context suffix
- subagents get a summarizing condenser by default unless disabled or replaced

Permission handling is explicit:

- `permission_mode` can be `always_confirm`, `never_confirm`, or
  `confirm_risky`
- absent `permission_mode` means inherit the parent confirmation policy
- the actual available tools come from the subagent definition's `tools` list

The inspected built-in subagents include:

- `general-purpose`: `terminal`, `file_editor`, `task_tracker`
- `code-explorer`: `terminal` only, instructed to use read-only commands
- `bash-runner`: `terminal` only, instructed to run commands and summarize
  results rather than dump raw output
- `web-researcher`: browser tooling plus fetch/Tavily MCP servers, instructed
  to cite sources and avoid side-effectful web actions

The built-ins do not include the task or delegate tools by default, so recursive
delegation is not enabled in the normal preset.

### Async and Background

OpenHands `delegate` supports parallelism by launching one thread per delegated
child task and joining all threads. It is concurrent but still blocking from the
parent tool call's perspective.

OpenHands `task` is also blocking. It does not have opencode-style background
mode and does not have ZeroClaw-style polling records. If the parent needs
background behavior, that would have to be provided by a higher-level runner or
server, not by `TaskManager.start_task` itself.

Separately, OpenHands agent tool execution can run tool calls through a
`ParallelToolExecutor` with resource locks and a configurable concurrency
limit. That is generic tool parallelism, not specifically subagent background
execution.

### Context and History

OpenHands conversation state is an event log plus a view used to prepare LLM
messages. Agent execution converts the current view into LLM messages, runs the
agent, records actions/observations/events, and can stop for confirmation,
budget, iteration limit, error, or completion.

Context management is condenser-based:

- `LLMSummarizingCondenser` summarizes forgotten context
- it can trigger from token limits or event-count policies
- it preserves a configured initial segment and recent tail
- it can recover from context overflow by condensing and retrying

This is materially stronger than simple truncation. It also applies to
subagents by default, because the subagent registry attaches a default
summarizing condenser unless the definition says otherwise.

### Best Read of OpenHands' Design

OpenHands separates two use cases clearly:

- Use `delegate` when you need persistent named workers that can receive
  multiple assignments over time.
- Use `task` when you need a scoped subagent task with optional resume by task
  id.

The strongest design choices are file-defined subagent capabilities, explicit
permission mode, independent child histories, and default context condensation.

## Cross-Repository Comparison

| Axis | ZeroClaw | opencode | OpenHands |
| --- | --- | --- | --- |
| Public mental model | Two primitives: ephemeral spawn and named delegate | One primitive: task | Two surfaces: explicit delegate and task |
| Child identity | `spawn_subagent` keeps parent identity; `delegate` uses named configured agents | Child session runs selected subagent type | Child agent from registered factory or markdown definition |
| Parent history visibility | Child does not see parent history unless included in prompt/context | Child does not see parent transcript unless resuming its own session | Child receives task prompt in its own conversation, not parent transcript |
| Child history returned? | No | No | No |
| Parent receives | Tool result string or background JSON | XML-like task result/error | Observation text with task id/status/result |
| Tool permissions | Inherited policy for spawn; filtered parent tools for agentic delegate; delegate excluded from delegate child | Parent denials plus subagent capabilities; task/todowrite denied by default | Tools listed in subagent definition; confirmation policy explicit or inherited |
| Recursive delegation | `spawn_subagent` depth-1 cap; agentic delegate excludes `delegate` | `task` denied by default for subagents | Built-ins do not include task/delegate by default |
| Recursion configurability | `delegate` has `max_delegation_depth` in the target runtime profile; `spawn_subagent` cap is hard-coded | Permission-configurable by allowing `task`; no numeric depth cap found | Tool-list configurable by adding task/delegate/workflow tools; no numeric depth cap found |
| Parallelism | `delegate.parallel` runs multiple agents and waits for all | Multiple task tool calls can be emitted in one model message; runtime handles them | `delegate` launches one thread per child task and waits for all |
| Background | `delegate.background` returns `task_id`; parent must check/cancel | Experimental background auto-injects result later | No task/delegate-level background mode in inspected SDK |
| Persistence/resume | Background result files; spawn sessions are ephemeral | `task_id` resumes child session | `task.resume` resumes persisted task conversation; `delegate` children remain live in executor |
| Context management | Tool result truncation and whole-turn trimming | Rich message parts plus compaction and tool-output pruning | Event log/view plus summarizing condenser |

## Recursion Limits

The repositories use two different safety models for recursive subagents:

- numeric depth counters
- tool/permission gating

ZeroClaw is the only inspected system with an explicit numeric delegation-depth
knob. `spawn_subagent` is hard-capped at one child level: a top-level agent can
spawn a subagent, but that subagent's own `spawn_subagent` call is refused by
the `is_subagent_caller` guard. This cap is not configurable as a depth number,
although the tool can still be allowed or denied by risk profile.

ZeroClaw `delegate` has a separate depth counter. The delegate tool starts at
depth `0`, child delegate constructions increment depth, and the target agent's
runtime profile resolves `max_delegation_depth`. If no valid profile value is
found, the fallback maximum is `3`. The check is `current_depth >= max_depth`,
so with the default, depth `0`, `1`, and `2` can delegate, while depth `3` is
blocked. In ordinary agentic delegation, ZeroClaw also filters `delegate` out of
the child tool registry, so recursion only exists where a child actually has a
delegate tool available.

opencode does not expose a numeric recursion depth for the `task` tool in the
inspected source. Instead, recursion is permission-gated. When a subagent
session is created, opencode adds a default deny for `task` unless the selected
subagent's own permission rules explicitly include `task`. Therefore built-in
subagents are non-recursive by default, but a custom subagent can be made
recursive by granting the `task` permission. That gives configurability, but not
a built-in "maximum depth N" safety guard.

OpenHands also uses tool-list gating rather than a numeric depth counter. The
built-in markdown subagents list tools such as `terminal`, `file_editor`,
`task_tracker`, or browser tools; they do not include `TaskToolSet`, `delegate`,
or `workflow`, so they cannot recursively launch subagents by default. A custom
subagent definition can include delegation-capable tools, but the inspected SDK
does not provide a built-in max recursion depth knob. The practical bounds are
tool availability, confirmation policy, max iterations, budget, and workflow
concurrency limits.

For decomposer design, numeric depth and tool gating solve different problems.
Tool gating is the first safety layer: most subagents should simply not have a
delegation tool. Numeric depth is still useful as a second safety layer for the
few profiles that are intentionally allowed to recursively decompose work.

## Spawn vs Delegate: The Conceptual Difference

`spawn_subagent` is useful when the child is the same agent in a clean room. It
answers: "Can I run focused reasoning or investigation in a fresh context and
receive only the conclusion?"

`delegate` is useful when the child is a distinct worker. It answers: "Can I
route this work to a named specialist with its own model, tools, policy,
lifecycle, and possibly background execution?"

opencode chooses not to expose this distinction. Its `task` tool covers both
"spawn a fresh worker" and "delegate a task" through subagent type and task id.
OpenHands keeps both styles, but uses clearer names: explicit `spawn` creates
named workers, while `task` launches scoped subagent work.

For decomposer design, `spawn` is redundant if every subtask can be represented
as a typed task with optional resume. `spawn` becomes useful only if the system
needs long-lived named workers, staged initialization, or persistent specialist
state before tasks are assigned.

## Background Execution: Responsibility Model

Background work has two different UX contracts:

- ZeroClaw returns a task id and tells the parent to call `check_result`. This
  makes the parent/session responsible for remembering to retrieve or cancel
  the task. It is simple and durable, but easy for the parent to forget.
- opencode returns a running task result and later injects the final result into
  the parent session. This makes the runtime responsible for delivery and tells
  the parent not to poll.

The opencode model is better for interactive agents because it prevents wasteful
polling and duplicated work. The ZeroClaw model is acceptable for batch systems
or CLIs where explicit task status commands are expected.

OpenHands' inspected `task` and `delegate` implementations are blocking. They
avoid the background responsibility problem by not exposing background mode at
that layer.

## Context Management Lessons

All three systems avoid the worst context failure mode: appending a full child
transcript to the parent. They differ in how they manage the remaining context.

ZeroClaw relies on bounded tool result strings and trimming. This is simple but
can lose information mechanically.

opencode stores richer message parts and has a compaction path that can
summarize older conversation while preserving recent turns. It also prunes old
tool outputs.

OpenHands uses an event log plus condenser. Subagents get summarizing
condensers by default, so long child runs can manage their own context without
spilling raw details into the parent.

For the decomposer, the parent should store:

- original user objective
- decomposition decisions
- child task specs
- compact child results
- execution status and errors

The parent should not store:

- every child reasoning step
- every child tool call
- raw command output unless the child result says it is essential
- full logs already available as artifacts

## Recommendations for the Decomposer

1. Use a single `task`-like primitive for v1 unless there is a concrete need
   for long-lived named workers. This follows opencode and OpenHands `task`.

2. Keep child history separate. The parent should receive a bounded result
   object, not a transcript. Include fields like status, summary, evidence,
   changed files, tests run, errors, and follow-up questions.

3. Make subagent prompts self-contained. A child should not rely on hidden
   parent context. The parent should pass the relevant files, logs, constraints,
   and expected output contract.

4. Add explicit subagent tool profiles. Examples:
   - explorer: read/search only
   - executor: shell/test focused
   - editor: file edits plus verification
   - researcher: web/documentation only

5. Disable recursive delegation by default. Allow it only through an explicit
   policy and depth budget. Prefer both controls: omit delegation tools from
   ordinary subagents, and apply a numeric maximum depth for the rare recursive
   profiles.

6. Prefer resumable task ids over separate spawn/delegate state if the main
   need is continuation. Use explicit `spawn` only if named workers need to be
   initialized before assignment or reused across multiple tasks.

7. If background execution is implemented, prefer opencode-style automatic
   completion delivery. Polling should be reserved for batch mode or explicit
   task-status UIs.

8. Add first-class compaction/condensation. Truncation is not enough for
   decomposer state. The decomposer should preserve decisions and final child
   findings while dropping raw child traces.

9. Collect parallel results deterministically. If multiple children run at
   once, wait for all by default and return ordered per-child results with
   explicit failures. Add racing or winner-selection only as a separate policy.

10. Track child metrics separately from parent metrics. ZeroClaw and OpenHands
    both treat delegated usage as attributable child work; the decomposer should
    do the same for cost and quality analysis.

## Practical Transfer Matrix

| Feature to copy | Best source | Why |
| --- | --- | --- |
| Simple subagent API | opencode `task` | One tool is easier for the parent model to learn |
| Explicit typed specialists | OpenHands markdown agents | Tools, model, budget, permission, and condenser are declarative |
| Background UX | opencode | Runtime injects result; parent does not poll |
| Durable polling fallback | ZeroClaw `delegate.background` | Useful for CLI/batch operation and crash recovery |
| Parallel fan-out | ZeroClaw/OpenHands | Both collect all child results with per-child status |
| Context isolation | All three | Child transcripts stay out of parent history |
| Context compaction | OpenHands/opencode | Summarization beats raw truncation |
| Recursion safety | ZeroClaw/opencode/OpenHands | Recursive delegation is blocked by default or omitted from subagent tools |

## Bottom Line

The best decomposer design is closest to opencode's public shape and
OpenHands' configuration model:

- expose one `task` primitive first
- back it with typed subagent profiles
- keep each child in a separate conversation/state object
- return compact structured results
- add resume ids
- make background completion automatic if background exists
- use summarizing context management before truncation

ZeroClaw's `spawn_subagent` is valuable as a clean-context same-agent fork, but
for decomposer purposes it should not be a separate public primitive unless the
decomposer has a concrete need for "same identity, fresh scratchpad" behavior.
