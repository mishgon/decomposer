# Decomposer Experiment on Claw-Eval

## Motivation

We want to test a hierarchical ReAct agentic pattern:

```text
Decomposer observes:
  main task + previous Executor subtasks and reports

Decomposer acts:
  either delegate the next natural-language subtask to Executor
  or produce the final response

Executor observes:
  a delegated subtask

Executor acts:
  run its own lower-level ReAct/tool-use loop in the environment
  then return a compact report to Decomposer
```

The Decomposer is a high-level agent. It receives the main task and can only
delegate natural-language subtasks to an Executor. It cannot call environment
tools directly.

The Executor is a lower-level agent. Given one subtask, it can interact with the
environment, use tools, inspect files/services, and then return a compact report.

The research hypothesis is:

> A strong Decomposer can manage a weaker Executor so that they solve agentic
> tasks better than the weak Executor acting alone.

This is different from static planning. The Decomposer does not need to predict
the whole solution upfront. It is itself a ReAct-style policy: observe the main
task and previous reports, act by delegating the next subtask, observe the new
report, then either continue or produce the final response.

## Benchmark

Use Claw-Eval, preferably the `general` split first.

Claw-Eval is suitable because many tasks are raw workflow requests rather than
step-by-step instructions. It also evaluates full trajectories, tool/service
side effects, safety constraints, and final task completion.

Avoid multimodal tasks in the first experiment unless the selected Executor is
known to support images/video/documents reliably. The goal of the first pass is
to test hierarchical task decomposition, not multimodal perception.

## Systems To Compare

### 1. Flat Agent Baselines

Models of different sizes (try Qwen3.5-0.8B/2B/4B/9B/27B and Qwen3.6-27B) act in a simple ReAct loop:

```text
ReAct loop:
  observe Claw-Eval task and environment/tool feedback
  act directly in the Claw-Eval environment
  continue until it returns the final answer or task state
```

### 2. Hierarchical ReAct (Decomposer + Executor)

A large model (try Qwen3.5-27B or Qwen3.6-27B) acts as Decomposer. A small model (try Qwen3.5-0.8B/2B/4B/9B) acts as Executor.

```text
Decomposer loop:
  observe Claw-Eval task and previous subtasks & reports
  act by delegating the next subtask or returning a final response

Executor loop for each delegated subtask:
  observe subtask
  act with Claw-Eval tools/environment
  observe tool/environment feedback
  return compact report
```

The Executor should use the same low-level scaffold, tool access, environment
access, stopping criteria, retry budget, and prompting style as the flat
baseline where possible. The only difference should be that it receives a
subtask from Decomposer instead of the original full task.

## Expected Outcomes

The target outcome is that a large Decomposer with a small Executor matches or
exceeds larger flat-agent baselines.

The hierarchy should help if the flat small agents fail because they lose track
of the global workflow, misses safety constraints, mixes up intermediate state,
or fails to verify completion.

The hierarchy may not help if failures are dominated by low-level tool-use
errors that the small Executor cannot perform even for simple subtasks.

This distinction is important. A negative result is still useful if the logs
show whether the bottleneck is high-level task management or low-level execution.

## Conversation Summary

The discussion framed the main OCC goal as building state-of-the-art small
language models for agentic tasks. A central motivation is that standard
agentic systems, such as Claude Code-style ReAct loops, accumulate more and
more context over time; this eventually hurts quality, and the problem is even
more severe for SLMs. Delegating subtasks to subagents is proposed as a way to
redistribute context hierarchically, so that each agent works with a smaller,
task-specific context.

The longer-term direction is role specialization among SLMs: separate models
for QA, tool calling, task decomposition, subagent management, and possibly
other agentic skills. The immediate research question is narrower: can a strong
manager, initially proxied by a large Qwen3.6-27B model, make small Qwen
subagents perform better than a single flat SLM agent? If this works, the
manager behavior can later be distilled into a specialized SLM.

The manager should not receive raw tool outputs or the full environment
context. Instead, the task context should be treated as an environment that
lower-level executors can inspect through tools. The manager's only effective
tool is delegation: it sends one natural-language subtask to a subagent and
receives a report describing success, failure, and relevant artifacts. This
should be a reactive loop rather than an upfront static plan: delegate one
subtask, observe the report, then decide the next subtask or final answer.

The discussion also clarified that the manager may know which types of
subagents are available and what they are good at. In the ideal version, it
could choose among specialized QA, tool-calling, execution, or even
manager-style agents. For the first experiment, however, the setup should stay
simple: one manager and one class of weaker Qwen executors, such as
Qwen3.5-0.8B, 2B, 4B, and 9B.

Claw-Eval and Toolathlon were identified as useful benchmarks because small
models should struggle on them, making it possible to test whether a large
manager can compensate. The initial run should be a proof of concept on a very
small number of tasks, debugged manually before scaling to roughly ten tasks or
more. Some Claw-Eval tasks require scraping credentials such as `SERP_DEV_KEY`,
and a small number require sandbox/container support. The judge model should be
kept fixed across baselines; using a large internal inference model was
considered acceptable to avoid spending external API budget.

Early implementation work found that the small Qwen executors initially did not
call tools because of tool-use flag handling, but this was fixed. After that,
tool calling started working both for the flat baseline and the decomposer
setup. The next bottleneck became executor reporting: smaller models,
especially 2B and 4B, could call tools but often failed to transition from tool
use to a final report, producing empty text or continuing to call tools. The 9B
executor was able to call tools and produce usable reports.

The group agreed that the manager should always receive some report. If an
executor fails to produce one, the system should return an empty or synthetic
failure report rather than hiding the failure. This behavior also appears to
help the manager adapt by issuing smaller subtasks. The remaining debugging
questions are whether the Qwen chat template is being used and parsed
correctly, whether final answers should appear in `content` after tool use, and
whether executor thinking should be enabled. Since the goal is an initial
quality proof of concept rather than token efficiency, enabling reasoning for
executors is considered acceptable, but it needs careful handling in the chat
template and parser.

## Experiment Update: Executor Prompt and Manager Tool Guidance

Date: 2026-06-25.

This run tested two decomposer ablations against flat Qwen3.5 baselines on
eight general Claw-Eval tasks: T112, T116, T118, T124, T114, T120, T126, and
T128. All runs used one trial per task. The manager was Qwen3.6-27B and the
executor was Qwen3.5-0.8B, 2B, 4B, or 9B. Judging used Qwen3.6-27B.

The implemented ablations were:

1. `transitional_retry`: executor prompt mode `flat_subtask`, executor thinking
   off, manager thinking off, strict report mode, no structured repair or
   report-as-tool, `executor_min_tool_calls=1`, and
   `retry_transitional_tool_text=true`.
2. `strict_manager_tools`: same as above, plus manager prompt guidance listing
   the exact executor tool names for the current task and instructing the
   manager to delegate only subtasks solvable with those tools.

The new configs are:

- `configs/experiments/config_decomposer_transitional_retry.yaml`
- `configs/experiments/config_decomposer_strict_manager_tools.yaml`

Focused tests passed:

```text
pytest tests/test_decomposer_prompts.py tests/test_decomposer.py tests/test_vllm_config.py
```

### Aggregate Scores

Scores are mean task scores over the eight tasks. Parentheses show pass count.

| Executor | Flat baseline | Decomp + transitional retry | Decomp + strict manager tools |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 0.224 (0/8) | 0.254 (0/8) | 0.318 (0/8) |
| Qwen3.5-2B | 0.217 (0/8) | 0.246 (0/8) | 0.266 (0/8) |
| Qwen3.5-4B | 0.332 (1/8) | 0.356 (0/8) | 0.404 (0/8) |
| Qwen3.5-9B | 0.651 (5/8) | 0.375 (0/8) | 0.348 (0/8) |

The best decomposer configuration in this run was Qwen3.6-27B manager +
Qwen3.5-4B executor with strict manager tool guidance, at 0.404 average score.
This improves over the flat 4B baseline by +0.072, but it still does not pass
any of the eight tasks. The flat 9B baseline remains much stronger at 0.651 and
5/8 passes.

### Per-Task Best Scores

This table compares the best score within each family.

| Task | Flat best | Transitional best | Strict manager best | Main observation |
|---|---:|---:|---:|---|
| T112 expense email | 0.844 | 0.382 | 0.386 | Flat 9B solves much more of the task. |
| T116 ticket KB | 0.200 | 0.440 | 0.433 | Decomposition helps; flat baselines fail. |
| T118 customer follow-up | 0.256 | 0.298 | 0.480 | Strict 4B improves, but manager can still stop too early. |
| T124 todo/calendar | 0.914 | 0.365 | 0.375 | Decomposer runs repeatedly miss the Mar 30 full-day trip conflict. |
| T114 meeting prep | 0.820 | 0.410 | 0.402 | Strict 9B failed by using wrong 2024 dates. |
| T120 inventory | 1.000 | 0.400 | 0.400 | Flat 9B remains far ahead. |
| T126 action items | 0.976 | 0.440 | 0.440 | Decomposer extracts useful pieces but loses final completeness. |
| T128 ticket assignment | 0.200 | 0.381 | 0.368 | Decomposition helps routing tasks where flat baselines are weak. |

### Decomposer Health Metrics

Average per eight-task run:

| Variant | Executor | Delegations | Executor turns | Executor tool calls | Empty visible executor responses | Transitional retries |
|---|---:|---:|---:|---:|---:|---:|
| transitional_retry | 0.8B | 4.75 | 13.38 | 5.00 | 6.38 | 0.25 |
| transitional_retry | 2B | 4.25 | 12.00 | 6.88 | 6.25 | 0.12 |
| transitional_retry | 4B | 4.62 | 16.50 | 15.25 | 5.25 | 2.25 |
| transitional_retry | 9B | 4.38 | 13.12 | 13.25 | 2.75 | 1.25 |
| strict_manager_tools | 0.8B | 7.25 | 20.75 | 8.62 | 8.50 | 0.88 |
| strict_manager_tools | 2B | 5.88 | 15.38 | 11.00 | 6.50 | 0.25 |
| strict_manager_tools | 4B | 3.38 | 13.00 | 16.88 | 2.88 | 2.38 |
| strict_manager_tools | 9B | 4.00 | 11.38 | 11.50 | 2.38 | 0.88 |

Strict manager tool guidance reduced obviously invalid delegations and helped
the 4B executor most. It also made 0.8B and 2B attempt more tool work, but weak
executors still produced many empty visible post-tool responses. For 0.8B, the
strict manager variant spent 522k total tokens across only eight tasks and still
had 8.5 empty visible executor responses per task on average.

### Failure Analysis

The main bottleneck is no longer only raw tool calling. Flat small agents and
decomposer executors can call tools. The harder problems are:

1. Weak executors still fail to turn tool observations into reliable compact
   reports. Transitional retry helps when the visible text says "I will fetch"
   but no tool call was emitted, but it does not fix blank post-tool reports.
2. The manager can reduce the ceiling even with a stronger executor. In strict
   9B T114, the manager delegated calendar searches for 2024-01-16/17/18 and
   then the whole 2024 year, while the task's mock date is 2026-03-26 and the
   correct "tomorrow" is 2026-03-27. The executor correctly followed the wrong
   date subtasks, so the final answer claimed there were no meetings.
3. The manager sometimes submits an unfinished planning sentence as the final
   answer. In strict 9B T118, after one CRM delegation the manager wrote that it
   needed to check email records and draft follow-ups, but called
   `submit_final_answer` instead of delegating the remaining work. The score was
   0.200.
4. The strict tool-name guidance helps with illegal subtasks such as shell or
   file-system requests, but it does not teach the manager when enough evidence
   has been collected or how to validate final task completion.

### Next Steps

1. Keep `strict_manager_tools + flat_subtask` as the main candidate for the 4B
   executor, then rerun with multiple seeds/trials before drawing conclusions.
2. Add manager final-answer gating. If the manager final answer contains
   planning language such as "I need to" or "let me check", or if required
   artifacts are missing, reject the final answer and force another delegation
   or a concrete final response.
3. Add task-date grounding. The manager should not infer dates from model
   priors. It should receive `environment.mock_today` when available, or be
   required to delegate a date-discovery subtask before resolving
   "today/tomorrow". T114 is the clearest regression.
4. Test `executor_min_tool_calls=0` under the strict manager prompt. The current
   runs still force at least one executor tool call, which may be wrong for
   report-only subtasks or synthesis subtasks after evidence has already been
   gathered.
5. Test report-as-tool or structured report only after the manager gating
   ablation. The current results show that manager finalization is a separate
   failure mode from executor blank reports.
6. Add per-subtask completion criteria to manager delegations. For example,
   "return the exact customer IDs and saved draft IDs" is safer than "review
   email records" because the manager can verify whether the report contains the
   artifacts needed for final scoring.
7. Operational note: a strict 50 GB GPU memory cap is not feasible for the
   Qwen3.6-27B manager in this setup. vLLM loaded the manager weights at about
   51.1 GiB before executor or KV-cache overhead.

## Experiment Update: Report Wrapper and Min-Tool-Calls Ablation

Date: 2026-06-25.

This run tested the earlier report-generation executor prompt again, now with
strict manager tool guidance and transitional tool-text retry enabled. The goal
was to check whether adding the flat baseline system prompt/date context and
changing `executor_min_tool_calls` would make the Executor reports reliable.

Setup:

- Decomposer: Qwen3.6-27B.
- Executors: Qwen3.5-0.8B, 2B, 4B, and 9B.
- Both manager and executor thinking disabled.
- Executor system prompt is built with the same `build_system_prompt` path as
  the flat baseline, so `environment.mock_today` is visible to the Executor.
- Manager still uses the decomposer coordinator prompt, not the flat baseline
  system prompt, so the manager is not directly date-grounded.
- `manager_valid_tool_guidance=true`.
- `retry_transitional_tool_text=true`.
- No structured repair and no report-as-tool.
- Two executor min-tool-call variants:
  - `report-wrapper min1`: `executor_min_tool_calls=1`.
  - `report-wrapper min0`: `executor_min_tool_calls=0`.

The new config is:

- `configs/experiments/config_decomposer_report_wrapper_min0_transitional_strict_tools.yaml`

Focused tests passed:

```text
/home/jovyan/.mlspace/envs/sukhorukov_decomposer/bin/python -m pytest tests/test_decomposer_prompts.py tests/test_decomposer.py tests/test_vllm_config.py
```

### Aggregate Scores

Scores are mean task scores over the same eight tasks as the previous table.
Parentheses show pass count.

| Executor | Flat | Decomp transitional | Decomp strict-tools | Report-wrapper min1 | Report-wrapper min0 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 0.224 (0/8) | 0.254 (0/8) | 0.318 (0/8) | 0.268 (0/8) | 0.247 (0/8) |
| Qwen3.5-2B | 0.217 (0/8) | 0.246 (0/8) | 0.266 (0/8) | 0.200 (0/8) | 0.200 (0/8) |
| Qwen3.5-4B | 0.332 (1/8) | 0.356 (0/8) | 0.404 (0/8) | 0.216 (0/8) | 0.254 (0/8) |
| Qwen3.5-9B | 0.651 (5/8) | 0.375 (0/8) | 0.348 (0/8) | 0.363 (0/8) | 0.302 (0/8) |

The report-wrapper prompt did not improve the hierarchy. It is worse than the
best previous decomposer setting (`strict_manager_tools + flat_subtask`) for
every executor size. It only beats the flat baseline for 0.8B, and only by
+0.044 for min1 and +0.023 for min0. For 4B and 9B it is much worse than the
flat baseline.

### Full Per-Task Table

| Variant | Executor | T112 | T116 | T118 | T124 | T114 | T120 | T126 | T128 | Avg | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat | 0.8B | 0.200 | 0.200 | 0.256 | 0.340 | 0.200 | 0.200 | 0.200 | 0.200 | 0.224 | 0/8 |
| flat | 2B | 0.200 | 0.200 | 0.200 | 0.340 | 0.200 | 0.200 | 0.200 | 0.200 | 0.217 | 0/8 |
| flat | 4B | 0.200 | 0.200 | 0.200 | 0.480 | 0.200 | 0.200 | 0.976 | 0.200 | 0.332 | 1/8 |
| flat | 9B | 0.844 | 0.200 | 0.256 | 0.914 | 0.820 | 1.000 | 0.976 | 0.200 | 0.651 | 5/8 |
| decomp transitional | 0.8B | 0.200 | 0.229 | 0.200 | 0.200 | 0.410 | 0.393 | 0.200 | 0.200 | 0.254 | 0/8 |
| decomp transitional | 2B | 0.200 | 0.200 | 0.200 | 0.358 | 0.410 | 0.200 | 0.200 | 0.200 | 0.246 | 0/8 |
| decomp transitional | 4B | 0.361 | 0.422 | 0.298 | 0.344 | 0.393 | 0.206 | 0.440 | 0.381 | 0.356 | 0/8 |
| decomp transitional | 9B | 0.382 | 0.440 | 0.229 | 0.365 | 0.386 | 0.400 | 0.433 | 0.367 | 0.375 | 0/8 |
| decomp strict-tools | 0.8B | 0.200 | 0.397 | 0.200 | 0.375 | 0.370 | 0.400 | 0.284 | 0.315 | 0.318 | 0/8 |
| decomp strict-tools | 2B | 0.200 | 0.200 | 0.200 | 0.365 | 0.402 | 0.232 | 0.200 | 0.331 | 0.266 | 0/8 |
| decomp strict-tools | 4B | 0.379 | 0.426 | 0.480 | 0.358 | 0.402 | 0.393 | 0.440 | 0.351 | 0.404 | 0/8 |
| decomp strict-tools | 9B | 0.386 | 0.433 | 0.200 | 0.361 | 0.200 | 0.394 | 0.440 | 0.368 | 0.348 | 0/8 |
| report-wrapper min1 | 0.8B | 0.200 | 0.200 | 0.220 | 0.326 | 0.402 | 0.400 | 0.200 | 0.200 | 0.268 | 0/8 |
| report-wrapper min1 | 2B | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0/8 |
| report-wrapper min1 | 4B | 0.206 | 0.200 | 0.229 | 0.200 | 0.200 | 0.212 | 0.284 | 0.200 | 0.216 | 0/8 |
| report-wrapper min1 | 9B | 0.382 | 0.200 | 0.402 | 0.319 | 0.394 | 0.386 | 0.433 | 0.390 | 0.363 | 0/8 |
| report-wrapper min0 | 0.8B | 0.218 | 0.231 | 0.220 | 0.305 | 0.200 | 0.386 | 0.217 | 0.200 | 0.247 | 0/8 |
| report-wrapper min0 | 2B | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0/8 |
| report-wrapper min0 | 4B | 0.200 | 0.200 | 0.298 | 0.252 | 0.200 | 0.397 | 0.284 | 0.200 | 0.254 | 0/8 |
| report-wrapper min0 | 9B | 0.368 | 0.200 | 0.220 | 0.270 | 0.200 | 0.400 | 0.433 | 0.330 | 0.302 | 0/8 |

### Report and Protocol Health

| Variant | Executor | Avg delegations | Avg exec turns | Avg env tool calls | Avg empty visible | Avg retries | Natural/synthetic reports | Avg manager protocol-error tool results | Runs with multiple-control failure |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| report-wrapper min1 | 0.8B | 6.62 | 27.00 | 12.38 | 9.75 | 4.12 | natural 42, synthetic 11 | 0.62 | 0/8 |
| report-wrapper min1 | 2B | 7.12 | 21.62 | 7.50 | 14.25 | 0.12 | natural 1, synthetic 56 | 0.50 | 0/8 |
| report-wrapper min1 | 4B | 7.00 | 28.00 | 28.88 | 11.00 | 4.25 | natural 36, synthetic 20 | 0.25 | 0/8 |
| report-wrapper min1 | 9B | 5.12 | 17.75 | 19.38 | 3.62 | 2.38 | natural 32, synthetic 9 | 0.38 | 0/8 |
| report-wrapper min0 | 0.8B | 7.62 | 30.75 | 12.75 | 10.62 | 4.62 | natural 44, synthetic 17 | 0.75 | 0/8 |
| report-wrapper min0 | 2B | 7.62 | 22.62 | 10.38 | 15.00 | 0.00 | natural 0, synthetic 61 | 0.25 | 0/8 |
| report-wrapper min0 | 4B | 6.75 | 28.75 | 33.50 | 11.50 | 4.75 | natural 36, synthetic 18 | 19.25 | 0/8 |
| report-wrapper min0 | 9B | 4.88 | 17.50 | 14.50 | 3.38 | 2.62 | natural 29, synthetic 10 | 4.38 | 1/8 |

Important detail: `executor_min_tool_calls=0` did not create true report-only
delegations in these runs. Across both report-wrapper variants, every
delegation still used at least one environment tool. The likely reason is that
the report-wrapper user prompt still says "First use the available tools", and
the manager's strict-tool subtasks were all phrased as tool-use tasks.

### Failure Analysis

The report-wrapper prompt appears to hurt the exact transition we are trying to
fix. Compared with the flat-subtask executor prompt, it adds a separate
"compact report" format requirement and encourages more post-tool text
protocol behavior. The 2B executor is the clearest case: min1 produced only
1 natural report out of 57 delegations, and min0 produced 0 natural reports out
of 61 delegations. Both scored exactly 0.200 on every task.

The 9B executor can still solve local subtasks when the manager gives clean
instructions. For example, report-wrapper min1 with 9B scored 0.402 on T118,
0.386 on T120, 0.433 on T126, and 0.390 on T128. But the hierarchy still loses
to flat 9B because the manager introduces errors:

1. Date grounding is still a manager problem. In report-wrapper min0 9B on
   T114, the manager first delegated "tomorrow (2024-01-17)", then corrected to
   "current date is 2026-01-13, so tomorrow is 2026-01-14". The executor
   followed those wrong subtasks and correctly found no meetings. The flat-style
   date context reached the Executor, but not the Decomposer.
2. Manager control protocol can corrupt otherwise useful answers. In
   report-wrapper min0 9B on T124, the manager generated a strong conflict
   analysis, but emitted two `submit_final_answer` tool calls in the same turn:
   one with the answer and one empty. The runner rejected the turn repeatedly
   with "call exactly one Decomposer control tool per turn" and the task ended
   with timeout / multiple-control failure.
3. The manager can spend delegations on lookup loops and miss side-effect
   completion. In report-wrapper min0 9B on T118, it used all eight
   delegations mostly for CRM/email lookup; it did not get to reliable
   saved-draft creation, so draft credit was zero.
4. Transitional retry is useful but noisy. It recovers some "I will call tool"
   textual turns, but it also fires around report-like text and adds extra tool
   calls. It does not solve blank visible responses after tool observations.

### Conclusions

The best current decomposer candidate remains
`strict_manager_tools + flat_subtask`, not `report_wrapper`. Matching the
executor system prompt to the flat baseline was necessary for date/tool parity,
but it was not sufficient because the manager has its own date and protocol
failure modes.

Removing the minimum one-tool-call constraint is not a meaningful fix in this
prompt form. To test that idea properly, the executor prompt must stop saying
"First use the available tools" and instead say "Use tools when needed; if the
subtask is only synthesis, answer directly." However, most current manager
subtasks are still tool-retrieval subtasks, so the expected impact is small
unless the manager is allowed to delegate synthesis-only subtasks.

### Next Steps

1. Use `strict_manager_tools + flat_subtask + transitional_retry` as the main
   candidate, not report-wrapper.
2. Add date grounding to the Decomposer prompt. If `task.environment.mock_today`
   exists, include it directly in the manager system prompt. This is required
   before rerunning date-sensitive tasks such as T114 and T124.
3. Add manager control repair or stricter decoding for control tools. A useful
   first repair is: if a manager turn contains exactly one nonempty
   `submit_final_answer` plus extra empty `submit_final_answer` calls, accept
   the nonempty answer and log a protocol repair. Keep rejecting genuinely
   conflicting multiple control actions.
4. Add manager final-answer gating. Reject final answers that are empty,
   planning-only, or missing required artifacts such as saved draft IDs.
5. Add an executor-only subtask replay benchmark. Feed the exact manager
   subtasks from successful and failed decomposer traces to the executor outside
   the manager loop. This separates "subtask too hard" from "manager delegated
   the wrong subtask".
6. If testing min0 again, change the report-wrapper wording to make tool use
   optional. Otherwise the ablation does not actually exercise no-tool
   reporting behavior.
7. Only after manager date/protocol fixes, test report-as-tool or structured
   executor reporting. The current runs show that report formatting alone is
   not the bottleneck and can make small executors worse.

### Direct Comparison to Flat Executor Prompt

The report-wrapper prompt is worse than the flat-subtask executor prompt. The
fairest comparison is against `flat_subtask + retry + strict tools`, because it
uses the same strict manager tool guidance and transitional retry settings as
the report-wrapper runs. Against that baseline, report-wrapper changes score by:

- 0.8B: min1 -0.049, min0 -0.071.
- 2B: min1 -0.066, min0 -0.066.
- 4B: min1 -0.187, min0 -0.150.
- 9B: min1 +0.015, min0 -0.045.

The small 9B min1 gain is not meaningful enough to prefer report-wrapper,
because flat standalone 9B is still much stronger at 0.651 and 5/8 passes, while
report-wrapper 9B has 0/8 passes.

The `n/a` cell below is not the standalone flat 9B baseline. It is only the
older decomposer raw flat-subtask ablation without retry or strict manager tool
guidance, which was run for 0.8B/2B/4B but not for 9B. The standalone flat 9B
baseline was run and is included in the first row.

| Variant | 0.8B | 2B | 4B | 9B |
|---|---:|---:|---:|---:|
| Standalone flat baseline | 0.224 (0/8) | 0.217 (0/8) | 0.332 (1/8) | 0.651 (5/8) |
| Decomp raw flat-subtask | 0.200 (0/8) | 0.234 (0/8) | 0.240 (0/8) | n/a |
| Decomp flat-subtask + retry | 0.254 (0/8) | 0.246 (0/8) | 0.356 (0/8) | 0.375 (0/8) |
| Decomp flat-subtask + retry + strict tools | 0.318 (0/8) | 0.266 (0/8) | 0.404 (0/8) | 0.348 (0/8) |
| Report wrapper min1 | 0.268 (0/8) | 0.200 (0/8) | 0.216 (0/8) | 0.363 (0/8) |
| Report wrapper min0 | 0.247 (0/8) | 0.200 (0/8) | 0.254 (0/8) | 0.302 (0/8) |

Plotted table artifact:

- `research/figures/decomposer_executor_prompt_comparison.svg`
- `research/figures/decomposer_executor_prompt_comparison.csv`

## Experiment Update: Prompt-V2 Manager Date and Report-Contract Rerun

Date: 2026-06-25.

This run repeated the main `flat_subtask + retry` decomposer candidate after
changing the Decomposer prompt. The prompt now includes `environment.mock_today`
when available, a stricter final-answer checklist, and a proposal-aligned
delegation report contract. The report contract tells the manager to request
natural-language task reports and concrete artifacts, not raw tool dumps or
tool internals. Both variants used the flat-subtask Executor prompt, manager and
executor thinking disabled, `retry_transitional_tool_text=true`, retry limit 2,
`executor_min_tool_calls=1`, and max 8 delegations.

Two variants were run:

1. `prompt-v2 no-strict`: manager does not receive valid executor tool names.
2. `prompt-v2 strict-tools`: manager receives task-specific executor tool names
   and is told to delegate only subtasks solvable with those tools.

Trace roots:

- `traces/decomposer_prompt_v2_flat_subtask_retry_no_strict_qwen36_27b`
- `traces/decomposer_prompt_v2_flat_subtask_retry_strict_tools_qwen36_27b`

Operational note: the first launch attempt OOMed during vLLM warmup because the
default `max_num_seqs=1024` was too large with other GPU jobs present. The final
runs used `--vllm-max-model-len 32768`,
`--vllm-gpu-memory-utilization 0.75`, and `--max-num-seqs 64` for manager,
executor, and judge servers. This should not change the deterministic task
policy, but it is a serving-configuration difference from some earlier runs.

Focused tests passed before the rerun:

```text
/home/jovyan/.mlspace/envs/sukhorukov_decomposer/bin/python -m pytest tests/test_decomposer_prompts.py tests/test_decomposer.py tests/test_vllm_config.py
```

### Aggregate Scores

Scores are mean task scores over the same eight tasks. Parentheses show pass
count.

| Variant | 0.8B | 2B | 4B | 9B |
|---|---:|---:|---:|---:|
| Standalone flat baseline | 0.224 (0/8) | 0.217 (0/8) | 0.332 (1/8) | 0.651 (5/8) |
| Old flat-subtask + retry | 0.254 (0/8) | 0.246 (0/8) | 0.356 (0/8) | 0.375 (0/8) |
| Old flat-subtask + retry + strict tools | 0.318 (0/8) | 0.266 (0/8) | 0.404 (0/8) | 0.348 (0/8) |
| Prompt-v2 flat-subtask + retry | 0.223 (0/8) | 0.233 (0/8) | 0.348 (0/8) | 0.323 (0/8) |
| Prompt-v2 flat-subtask + retry + strict tools | 0.251 (0/8) | 0.286 (0/8) | 0.374 (0/8) | 0.366 (0/8) |

The prompt-v2 change did not improve the main aggregate result. The best new
run was prompt-v2 strict-tools with the 4B executor at 0.374, but the old
strict-tools 4B run remains higher at 0.404. Standalone flat 9B is still much
stronger at 0.651 and 5/8 passes.

### Full Per-Task Table

| Variant | Executor | T112 | T116 | T118 | T124 | T114 | T120 | T126 | T128 | Avg | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat | 0.8B | 0.200 | 0.200 | 0.256 | 0.340 | 0.200 | 0.200 | 0.200 | 0.200 | 0.224 | 0/8 |
| flat | 2B | 0.200 | 0.200 | 0.200 | 0.340 | 0.200 | 0.200 | 0.200 | 0.200 | 0.217 | 0/8 |
| flat | 4B | 0.200 | 0.200 | 0.200 | 0.480 | 0.200 | 0.200 | 0.976 | 0.200 | 0.332 | 1/8 |
| flat | 9B | 0.844 | 0.200 | 0.256 | 0.914 | 0.820 | 1.000 | 0.976 | 0.200 | 0.651 | 5/8 |
| prompt-v2 no-strict | 0.8B | 0.200 | 0.200 | 0.200 | 0.200 | 0.385 | 0.200 | 0.200 | 0.200 | 0.223 | 0/8 |
| prompt-v2 no-strict | 2B | 0.200 | 0.200 | 0.200 | 0.264 | 0.393 | 0.206 | 0.200 | 0.200 | 0.233 | 0/8 |
| prompt-v2 no-strict | 4B | 0.200 | 0.418 | 0.472 | 0.332 | 0.393 | 0.382 | 0.200 | 0.386 | 0.348 | 0/8 |
| prompt-v2 no-strict | 9B | 0.364 | 0.200 | 0.220 | 0.351 | 0.424 | 0.386 | 0.440 | 0.200 | 0.323 | 0/8 |
| prompt-v2 strict-tools | 0.8B | 0.200 | 0.200 | 0.220 | 0.239 | 0.387 | 0.206 | 0.225 | 0.331 | 0.251 | 0/8 |
| prompt-v2 strict-tools | 2B | 0.200 | 0.200 | 0.200 | 0.252 | 0.402 | 0.400 | 0.433 | 0.200 | 0.286 | 0/8 |
| prompt-v2 strict-tools | 4B | 0.379 | 0.440 | 0.220 | 0.400 | 0.363 | 0.386 | 0.433 | 0.368 | 0.374 | 0/8 |
| prompt-v2 strict-tools | 9B | 0.361 | 0.418 | 0.239 | 0.337 | 0.389 | 0.393 | 0.433 | 0.360 | 0.366 | 0/8 |

### Per-Task Best Scores

| Task | Flat best | Old no-strict best | Old strict best | New no-strict best | New strict best |
|---|---:|---:|---:|---:|---:|
| T112 | 0.844 | 0.382 | 0.386 | 0.364 | 0.379 |
| T116 | 0.200 | 0.440 | 0.433 | 0.418 | 0.440 |
| T118 | 0.256 | 0.298 | 0.480 | 0.472 | 0.239 |
| T124 | 0.914 | 0.365 | 0.375 | 0.351 | 0.400 |
| T114 | 0.820 | 0.410 | 0.402 | 0.424 | 0.402 |
| T120 | 1.000 | 0.400 | 0.400 | 0.386 | 0.400 |
| T126 | 0.976 | 0.440 | 0.440 | 0.440 | 0.433 |
| T128 | 0.200 | 0.381 | 0.368 | 0.386 | 0.368 |

Date grounding helped the intended class of failures: T114 no longer uses the
2024 dates seen in earlier runs. New no-strict 9B reached 0.424 on T114, and
new strict 9B recovered from the old strict 9B score of 0.200 to 0.389. But
date grounding alone did not raise the aggregate score.

### Health Metrics

Average per eight-task run. `Natural/Synth` is the total count of natural
executor reports versus synthetic failure reports across the eight tasks.

| Variant | Executor | Deleg | Turns | Tools | Empty | Retries | Natural/Synth | Mgr protocol err avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| old no-strict | 0.8B | 4.75 | 13.38 | 5.00 | 6.38 | 0.25 | 13/25 | 0.00 |
| old no-strict | 2B | 4.25 | 12.00 | 6.88 | 6.25 | 0.12 | 11/23 | 0.00 |
| old no-strict | 4B | 4.62 | 16.50 | 15.25 | 5.25 | 2.25 | 26/11 | 0.25 |
| old no-strict | 9B | 4.38 | 13.12 | 13.25 | 2.75 | 1.25 | 28/7 | 0.25 |
| old strict-tools | 0.8B | 7.25 | 20.75 | 8.62 | 8.50 | 0.88 | 27/31 | 0.12 |
| old strict-tools | 2B | 5.88 | 15.38 | 11.00 | 6.50 | 0.25 | 22/25 | 0.38 |
| old strict-tools | 4B | 3.38 | 13.00 | 16.88 | 2.88 | 2.38 | 24/3 | 0.50 |
| old strict-tools | 9B | 4.00 | 11.38 | 11.50 | 2.38 | 0.88 | 27/5 | 0.12 |
| prompt-v2 no-strict | 0.8B | 4.88 | 13.88 | 5.50 | 8.25 | 0.00 | 6/33 | 0.12 |
| prompt-v2 no-strict | 2B | 6.50 | 17.88 | 11.50 | 9.25 | 0.12 | 16/36 | 0.25 |
| prompt-v2 no-strict | 4B | 5.12 | 17.62 | 20.50 | 5.75 | 2.25 | 29/12 | 0.00 |
| prompt-v2 no-strict | 9B | 4.50 | 15.25 | 14.00 | 2.25 | 2.25 | 26/10 | 0.12 |
| prompt-v2 strict-tools | 0.8B | 8.00 | 21.75 | 9.62 | 8.50 | 0.75 | 30/34 | 0.12 |
| prompt-v2 strict-tools | 2B | 7.50 | 20.62 | 8.50 | 10.62 | 0.12 | 18/42 | 0.00 |
| prompt-v2 strict-tools | 4B | 5.62 | 17.75 | 19.25 | 4.50 | 2.12 | 34/11 | 0.12 |
| prompt-v2 strict-tools | 9B | 5.25 | 16.12 | 18.50 | 3.75 | 1.50 | 32/10 | 0.12 |

Prompt-v2 made the manager more active. That is not always good: strict-tools
4B went from 3.38 to 5.62 delegations per task and from 24/3 to 34/11
natural/synthetic reports. The manager collected more evidence, but the extra
delegations also exposed more executor blank-report failures and used the
delegation budget on lookup loops.

### What Is Different From The Standalone Flat Baseline

These differences can impair fair comparison:

1. The standalone flat agent sees the original task and acts directly in the
   environment. The decomposer Executor sees manager-generated subtasks, so any
   manager error becomes an executor input error.
2. The Decomposer sees Executor reports, not raw environment tool outputs. If
   the Executor calls tools but returns a synthetic/empty report, the manager
   loses the evidence.
3. The decomposer loop has an explicit manager budget: max 8 delegations and
   15 manager turns. Flat runs do not have this manager bottleneck.
4. These decomposer runs still use `executor_min_tool_calls=1`, while the flat
   agent can stop naturally after answering. This can be wrong for synthesis or
   final-report subtasks.
5. Transitional retry is enabled in the decomposer Executor loop. It recovers
   some "I will call a tool" text turns, but it can also trigger extra tool
   calls when the text is actually report-like.
6. Strict-tools is not a pure proposal setting. The manager is still not using
   environment tools directly, but it receives a list of executor tool names,
   which gives it more environment-specific knowledge than the no-strict
   proposal-aligned variant.
7. The prompt-v2 serving run used capped vLLM concurrency/memory settings due
   to other GPU processes.

### Failure Analysis

The new prompt fixed one known problem but exposed the same broader bottlenecks:

1. Date grounding is necessary but insufficient. T114 improved because the
   manager now knows that `mock_today=2026-03-26`, so tomorrow is 2026-03-27.
   But T118 is date-sensitive while its YAML does not expose `mock_today`; the
   correct date only appears in the judge rubric/reference solution. In those
   traces, the Executor invented dates such as 2026-03-27 or 2026-07-16,
   causing incorrect customer eligibility calculations.
2. The Executor still drops visible reports after tool use. Prompt-v2 strict
   4B T118 is the clearest example: the manager correctly asked for a saved
   Gmail draft, but the Executor called `crm_get_customer` instead of
   `gmail_save_draft` and then returned no valid report twice. The manager
   eventually wrote a draft in the final answer, but no draft artifact existed,
   so draft credit was zero.
3. The manager now often decomposes into smaller lookup subtasks. This helps
   T112/T126-style extraction but can hurt side-effect tasks because all
   delegations get spent before actions such as saving drafts or assigning
   tickets are completed.
4. Strict-tools helps valid tool selection but does not solve task strategy.
   It improved prompt-v2 2B over no-strict and kept 4B/9B competitive, but it
   also encouraged tool-name-heavy retrieval subtasks and more lookup loops.
5. The report contract is aligned with the proposal, but it may be too weak as
   an operational guardrail. It asks for artifacts in reports, yet does not
   force the manager to verify that side-effect artifacts such as draft IDs or
   ticket assignments actually exist before submitting the final answer.

### Next Steps

1. Keep `flat_subtask + retry`, but do not treat prompt-v2 as a clear win. The
   strongest current aggregate remains the old strict-tools 4B result.
2. Add side-effect-aware final-answer gating. For tasks requiring saved drafts,
   ticket updates, to-do creation, or similar artifacts, reject final answers
   unless the manager has seen concrete artifact IDs in Executor reports.
3. Add explicit date context for all date-sensitive tasks, not only tasks that
   already define `environment.mock_today`. T118 should either expose
   `mock_today=2026-03-26` or include that date in the task prompt; otherwise
   the Executor can still invent dates while doing local calculations.
4. Add an executor-only replay suite from manager subtasks. Replay failed
   subtasks such as "save the Gmail draft for CUS-704" with the same Executor
   model to separate subtask difficulty from manager strategy.
5. Test optional tool use properly by changing the Executor prompt to "use
   tools when needed" and setting `executor_min_tool_calls=0`. The current
   prompt still strongly nudges every subtask toward tool use.
6. Consider a narrow report-as-tool ablation only for the Executor return, not
   for environment actions. The evidence still points to post-tool reporting as
   a bottleneck, especially for 0.8B/2B.
7. For fairer flat comparison, run multiple trials for the best decomposer
   candidates and rerun standalone flat with the same vLLM serving cap and
   retry settings.

Plotted table artifacts:

- `research/figures/decomposer_prompt_v2_comparison.svg`
- `research/figures/decomposer_prompt_v2_comparison.csv`
- `research/figures/decomposer_prompt_v2_per_task.csv`

## Experiment Update: Prompt-V2 Strict-Tools Budget-16 Rerun

Date: 2026-06-25.

This run tested whether the prompt-v2 strict-tools candidate was mainly
limited by the manager budget. The only intended behavioral budget changes were:

- `max_delegations`: 8 -> 16.
- `max_decomposer_turns`: 15 -> 32.

The rest of the candidate stayed aligned with the previous prompt-v2
strict-tools setup: Qwen3.6-27B manager, Qwen3.5 Executor, flat-subtask
Executor prompt, thinking disabled for both manager and executor,
`retry_transitional_tool_text=true`, retry limit 2, strict manager tool
guidance enabled, and `executor_min_tool_calls=1`.

Config:

- `configs/experiments/config_decomposer_prompt_v2_strict_tools_budget16.yaml`

Trace directories:

- `traces/decomposer_prompt_v2_strict_tools_budget16_qwen36_27b/Qwen_Qwen3.6-27B__Qwen_Qwen3.5-0.8B_26-06-25-18-15`
- `traces/decomposer_prompt_v2_strict_tools_budget16_qwen36_27b/Qwen_Qwen3.6-27B__Qwen_Qwen3.5-2B_26-06-25-18-28`

Focused tests passed before running:

```text
/home/jovyan/.mlspace/envs/sukhorukov_decomposer/bin/python -m pytest tests/test_decomposer_prompts.py tests/test_decomposer.py tests/test_vllm_config.py
```

Operational note: inline grading hung after the first 0.8B task, so the final
runs used trace-only execution with `--skip-grade`, followed by separate
`grade-batch` passes with Qwen3.6-27B as judge. All eight tasks were graded for
both Executor sizes.

### Aggregate Scores

Scores are mean task scores over the same eight tasks. Parentheses show pass
count.

| Variant | Budget | 0.8B | 2B | 4B | 9B |
|---|---|---:|---:|---:|---:|
| Standalone flat baseline | n/a | 0.224 (0/8) | 0.217 (0/8) | 0.332 (1/8) | 0.651 (5/8) |
| Old flat-subtask + retry + strict tools | 8 deleg / 15 turns | 0.318 (0/8) | 0.266 (0/8) | 0.404 (0/8) | 0.348 (0/8) |
| Prompt-v2 flat-subtask + retry + strict tools | 8 deleg / 15 turns | 0.251 (0/8) | 0.286 (0/8) | 0.374 (0/8) | 0.366 (0/8) |
| Prompt-v2 flat-subtask + retry + strict tools | 16 deleg / 32 turns | 0.349 (0/8) | 0.320 (0/8) | n/a | n/a |

Increasing the manager budget improved both small Executor runs:

- 0.8B improved from 0.251 to 0.349.
- 2B improved from 0.286 to 0.320.

However, pass rate stayed 0/8. The budget increase helps evidence collection
and some synthesis tasks, but it does not close the gap to standalone flat 9B,
which remains 0.651 with 5/8 passes.

### Full Per-Task Table

| Variant | Executor | T112 | T116 | T118 | T124 | T114 | T120 | T126 | T128 | Avg | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standalone flat | 0.8B | 0.200 | 0.200 | 0.256 | 0.340 | 0.200 | 0.200 | 0.200 | 0.200 | 0.224 | 0/8 |
| standalone flat | 2B | 0.200 | 0.200 | 0.200 | 0.340 | 0.200 | 0.200 | 0.200 | 0.200 | 0.217 | 0/8 |
| prompt-v2 strict, budget 8 | 0.8B | 0.200 | 0.200 | 0.220 | 0.239 | 0.387 | 0.206 | 0.225 | 0.331 | 0.251 | 0/8 |
| prompt-v2 strict, budget 8 | 2B | 0.200 | 0.200 | 0.200 | 0.252 | 0.402 | 0.400 | 0.433 | 0.200 | 0.286 | 0/8 |
| prompt-v2 strict, budget 16 | 0.8B | 0.200 | 0.418 | 0.249 | 0.285 | 0.436 | 0.394 | 0.433 | 0.373 | 0.349 | 0/8 |
| prompt-v2 strict, budget 16 | 2B | 0.200 | 0.200 | 0.249 | 0.326 | 0.410 | 0.400 | 0.433 | 0.343 | 0.320 | 0/8 |

The larger budget mainly helped tasks that benefit from additional independent
lookups or final synthesis:

- 0.8B gained strongly on T116, T120, T126, and T128.
- 2B gained on T124 and T128 while preserving T120/T126.
- T112 stayed at 0.200 for both models, so more manager budget did not fix the
  reimbursement matching workflow.
- T116 is unstable across Executor size: 0.8B reached 0.418, while 2B stayed at
  0.200 in this run.

### Health Metrics

Average per eight-task run. `Natural/Synth` is the total count of natural
Executor reports versus synthetic failure reports across the eight tasks.

| Variant | Executor | Deleg | Turns | Tools | Empty | Retries | Natural/Synth | Tokens/task |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| prompt-v2 strict, budget 8 | 0.8B | 8.00 | 21.75 | 9.62 | 8.50 | 0.75 | 30/34 | n/a |
| prompt-v2 strict, budget 8 | 2B | 7.50 | 20.62 | 8.50 | 10.62 | 0.12 | 18/42 | n/a |
| prompt-v2 strict, budget 16 | 0.8B | 12.38 | 30.88 | 13.25 | 10.50 | 0.38 | 55/44 | 100045 |
| prompt-v2 strict, budget 16 | 2B | 10.00 | 27.25 | 15.62 | 12.38 | 0.50 | 35/45 | 93709 |

The larger budget increased useful natural reports, especially for 0.8B
(30 -> 55 natural reports), but it also increased empty visible responses and
synthetic reports. Three of eight tasks hit the 16-delegation cap for each
Executor size, so some tasks are still budget-bound even after doubling the
limit.

### Interpretation

The budget hypothesis is partially supported: the old 8-delegation cap was too
tight for small Executors, and increasing it produced a clear aggregate gain.
But budget is not the primary reason the decomposer still fails to solve tasks.
The remaining bottlenecks are:

1. Executor report reliability. Empty visible responses still occur frequently
   after tool calls, so the manager often receives lossy or synthetic evidence.
2. Side-effect verification. More delegations collect more evidence, but do not
   guarantee that required artifacts such as drafts, ticket updates, or final
   assignments were actually created.
3. Manager loop quality. Additional budget is sometimes spent on repeated
   lookup subtasks instead of converting evidence into final actions.
4. Task-specific instability. T116 improved for 0.8B but not 2B, while T112 did
   not improve at all. This suggests the issue is not just capacity or budget;
   the manager/executor protocol still has task-level failure modes.

Next steps from this result:

1. Keep budget 16 for the next small-Executor decomposer candidate; budget 8 is
   probably too tight for 0.8B/2B.
2. Add side-effect-aware final-answer gating before increasing the budget again.
3. Test `executor_min_tool_calls=0` with the same budget 16 setting, because
   some synthesis/report subtasks should not be forced into a tool call.
4. Run the budget-16 setting for 4B and 9B only after fixing report/side-effect
   reliability; otherwise the larger run is likely to spend more GPU on the
   same failure mode.

## Evaluation Fix: Sidecar-Aware Decomposer Grading

Date: 2026-06-26.

The previous decomposer tables are not directly comparable to flat baselines.
The issue was in evaluation, not only in the decomposer policy: task graders
apply tool-use gates from `ToolDispatch` events, but decomposer environment
tool calls are stored in Executor sidecar traces. The manager trace contains
`delegation_end` records and audit snapshots, but zero environment
`tool_dispatch` events. As a result, every decomposer task was penalized as if
it had not used the required tools.

The evaluation path was changed so grading keeps the manager-visible
conversation for LLM judge prompts, but augments the grader `dispatches` list
with `ToolDispatch` events from each `delegation_end.sidecar_trace`.

Implementation:

- Added sidecar-aware grading load in `src/claw_eval/trace/reader.py`.
- Wired `grade`, `grade-batch`, and inline decomposer grading paths to use it.
- Added regression tests for sidecar dispatch merging and `grade-batch`.

Focused tests:

```text
/home/jovyan/.mlspace/envs/sukhorukov_decomposer/bin/python -m pytest tests/test_decomposer.py tests/test_grade_batch.py tests/test_decomposer_prompts.py tests/test_vllm_config.py
```

Result: 38 passed.

### Sidecar-Aware Regrade: Prompt-V2 Strict 9B

Existing trace regraded with `--force` and Qwen3.6-27B judge:

- `traces/decomposer_prompt_v2_flat_subtask_retry_strict_tools_qwen36_27b/Qwen_Qwen3.6-27B__Qwen_Qwen3.5-9B_26-06-25-15-08`

Comparison to standalone flat 9B:

| Variant | T112 | T116 | T118 | T124 | T114 | T120 | T126 | T128 | Avg | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standalone flat 9B | 0.844 | 0.200 | 0.256 | 0.914 | 0.820 | 1.000 | 0.976 | 0.200 | 0.651 | 5/8 |
| decomposer 9B, old grading | 0.361 | 0.418 | 0.239 | 0.337 | 0.389 | 0.393 | 0.433 | 0.360 | 0.366 | 0/8 |
| decomposer 9B, sidecar-aware grading | 0.844 | 0.928 | 0.312 | 0.692 | 0.844 | 1.000 | 0.976 | 0.708 | 0.788 | 5/8 |

This changes the main conclusion. With correct grading, the prompt-v2 strict
9B decomposer is not worse than the flat 9B baseline on this eight-task set; it
has the same pass count and a higher average score. Decomposition improves the
flat baseline failures T116 and T128, while the remaining failures are now
specific task-policy issues:

1. T118 still fails because of date grounding and missing saved Gmail drafts.
2. T124 misses the 2026-03-30 calendar range and falls below pass threshold.
3. T128 improves substantially but still misroutes some tickets.

Next steps after this fix:

1. Regrade the existing decomposer runs with sidecar-aware evaluation before
   drawing model-size or prompt conclusions from older tables.
2. Treat prior `0/8` decomposer pass counts as invalid unless they were
   produced after this evaluation fix.
3. Focus decomposer improvements on the remaining genuine failures: date
   context for T118, calendar horizon selection for T124, and routing/contact
   reasoning for T128.

## Full Sidecar-Aware Regrade of Existing Decomposer Runs

Date: 2026-06-26.

I regraded the existing main decomposer traces with the sidecar-aware grading
path. The pass used two local Qwen3.6-27B judge vLLM servers in parallel:
GPU0 on port 8000 and GPU1 on port 8001. All selected traces were regraded
with `grade-batch --force`; the already-corrected prompt-v2 strict-tools 9B
trace was left as-is to avoid appending a redundant judge result. Both vLLM
servers were stopped after the pass.

Artifacts:

- `research/figures/sidecar_regrade_aggregate.csv`
- `research/figures/sidecar_regrade_per_task.csv`
- `research/figures/sidecar_regrade_key_comparison.csv`
- `research/figures/sidecar_regrade_final_comparison.svg`
- `research/figures/sidecar_regrade_key_per_task.svg`

### Aggregate Table

Scores are mean task scores. Parentheses show pass count. Rows with four tasks
are ablations on a smaller task subset and should not be compared directly to
the eight-task flat baseline.

| Variant | Tasks | 0.8B | 2B | 4B | 9B |
|---|---:|---:|---:|---:|---:|
| standalone flat baseline | 8 | 0.224 (0/8) | 0.217 (0/8) | 0.332 (1/8) | 0.651 (5/8) |
| flat-exec prompt + retry + strict manager tools | 8 | 0.546 (3/8) | 0.405 (2/8) | 0.893 (7/8) | 0.725 (5/8) |
| flat-exec prompt + transitional retry | 8 | 0.385 (2/8) | 0.367 (2/8) | 0.579 (3/8) | 0.824 (7/8) |
| prompt-v2 flat-subtask + retry + strict tools | 8 | 0.336 (1/8) | 0.512 (3/8) | 0.809 (5/8) | 0.788 (5/8) |
| prompt-v2 flat-subtask + retry, no strict tools | 8 | 0.277 (1/8) | 0.294 (1/8) | 0.684 (6/8) | 0.640 (5/8) |
| prompt-v2 strict tools, budget16/turn32 | 8 | 0.669 (4/8) | 0.582 (3/8) | n/a | n/a |
| report prompt + retry + strict tools, min tool 1 | 8 | 0.454 (2/8) | 0.200 (0/8) | 0.253 (0/8) | 0.773 (6/8) |
| report prompt + retry + strict tools, min tool 0 | 8 | 0.365 (1/8) | 0.200 (0/8) | 0.395 (1/8) | 0.590 (3/8) |
| flat-exec prompt, min tool 0 | 4 | 0.236 (0/4) | 0.200 (0/4) | 0.202 (0/4) | n/a |
| flat-exec prompt, no retry baseline ablation | 4 | 0.200 (0/4) | 0.358 (1/4) | 0.358 (1/4) | n/a |

### Key Per-Task Comparison

| Variant | Avg | Pass | T112 | T116 | T118 | T124 | T114 | T120 | T126 | T128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat 9B | 0.651 | 5/8 | 0.844 | 0.200 | 0.256 | 0.914 | 0.820 | 1.000 | 0.976 | 0.200 |
| strict-manager 4B | 0.893 | 7/8 | 0.844 | 0.928 | 1.000 | 0.846 | 0.872 | 0.944 | 0.976 | 0.732 |
| transitional-retry 9B | 0.824 | 7/8 | 0.928 | 1.000 | 0.284 | 0.860 | 0.820 | 0.972 | 0.976 | 0.756 |
| prompt-v2 strict 4B | 0.809 | 5/8 | 0.916 | 1.000 | 0.256 | 0.976 | 0.716 | 0.972 | 0.976 | 0.660 |
| prompt-v2 strict 9B | 0.788 | 5/8 | 0.844 | 0.928 | 0.312 | 0.692 | 0.844 | 1.000 | 0.976 | 0.708 |
| report min1 9B | 0.773 | 6/8 | 0.928 | 0.200 | 0.776 | 0.704 | 0.848 | 0.944 | 0.976 | 0.808 |

### Interpretation

The sidecar-aware regrade invalidates the earlier conclusion that the
decomposer could not solve these tasks. Several decomposer variants now beat
the standalone flat 9B baseline on outcome metrics:

- Best average: flat-exec prompt + retry + strict manager tools with 4B
  Executor, 0.893 average and 7/8 passes.
- Best tied pass count: flat-exec prompt + transitional retry with 9B
  Executor, 0.824 average and 7/8 passes.
- Prompt-v2 strict-tools remains strong for 4B/9B, but it is not the best
  observed setting after the corrected evaluation.
- The budget16 prompt-v2 rerun materially helps small Executors: 0.8B reaches
  0.669 and 4/8 passes; 2B reaches 0.582 and 3/8 passes.
- The report-wrapper Executor prompt is unstable. It works for 9B with
  min-tool 1, but collapses for 2B and is weak for 4B in these traces.

This is not a compute-fair comparison against flat baseline: decomposer runs
use a Qwen3.6-27B manager plus the Executor and consume substantially more
tokens/wall time. The corrected result says decomposition can improve outcome
quality on this task set, not that it is more efficient.

The remaining genuine issues are task- and protocol-specific rather than
global tool-use failure:

1. T118 is still brittle for many candidates because date grounding and Gmail
   draft side effects are easy to miss.
2. T128 improves greatly under decomposition but often remains just below pass
   threshold because routing/contact reasoning is partially wrong.
3. T124 is strong in some variants but drops in prompt-v2 9B, so calendar
   horizon selection is still inconsistent.
4. Small Executors benefit from more budget, but they still need better
   report reliability and final-action verification.

Next steps:

1. Treat sidecar-aware grading as mandatory for all decomposer comparisons.
2. Use strict-manager-tools 4B and transitional-retry 9B as the current
   strongest baselines for follow-up trace analysis.
3. Run multi-seed repeats for the strongest candidates before claiming a stable
   model-size trend.
4. Add side-effect-aware final-answer checks for drafts, ticket changes,
   assignments, and calendar edits, then rerun prompt-v2 strict tools.
5. Report outcome and cost separately: average/pass count, total tokens, and
   wall time should all be shown in future tables.

## Configuration Update: Transitional Retry Disabled

Date: 2026-06-26.

The transitional tool-text retry is now disabled in the flat local vLLM
baseline and all decomposer experiment configs. Historical result rows and
trace directories that mention retry still describe already-run experiments,
but future flat/decomposer runs should not use this protocol repair. The
underlying ReAct compatibility option remains in code for old config support
and targeted low-level tests.

## No-Protocol Smoke Validation

Date: 2026-06-26.

Implemented the no-protocol comparison setup:

- Flat vLLM baseline keeps the comparison caps:
  `react.max_turns=352` and `react.max_environment_tool_calls=320`.
- Flat and Executor ReAct loops now default to no model-visible protocol
  repairs:
  `retry_empty_model_response=false`,
  `retry_missing_required_tool=false`,
  `retry_transitional_tool_text=false`.
- Decomposer control-tool corrections remain active for the manager control
  loop.
- Decomposer experiment configs set
  `executor_synthetic_failure_report=false`, so strict report extraction
  records real `missing_report` failures instead of fabricated Executor
  failure reports.
- `executor_min_tool_calls=0` now truly allows the first Executor turn to
  finish without forcing a tool call.

While smoke-testing, the local vLLM launcher exposed two environment-priority
bugs:

1. Flat launches updated generic `VLLM_BASE_URL` but not role-specific
   `VLLM_MODEL_BASE_URL` / `VLLM_MODEL_MODEL_ID`, so stale `.env` values could
   point the flat runner at the wrong port.
2. Flat config loading could be hijacked by stale `VLLM_DECOMPOSER_*` values.
   Decomposer role env now overrides `cfg.model` only when the loaded YAML has
   an `executor_model` block.

Targeted tests passed after the fixes:

```bash
pytest tests/test_vllm_process.py tests/test_vllm_config.py tests/test_react_loop.py tests/test_decomposer.py tests/test_flat_runner.py
```

Result: 49 passed.

Trace-only T124 smoke runs:

| Variant | Trace root | Outcome |
|---|---|---|
| flat Qwen3.5-0.8B | `traces/no_protocol_nosynth_smoke_26-06-26_flat_run_fixed2/.../T124_todo_calendar_conflict_1d222826.jsonl` | 1 turn, no tool calls, no protocol corrections |
| decomposer no strict tools, Qwen3.6-27B + Qwen3.5-0.8B | `traces/no_protocol_nosynth_smoke_26-06-26_decomposer_no_strict_fixed/T124_todo_calendar_conflict_a2f52524.jsonl` | 16 delegations, 16 Executor `missing_report`, 0 Executor tool calls |
| decomposer strict tools, Qwen3.6-27B + Qwen3.5-0.8B | `traces/no_protocol_nosynth_smoke_26-06-26_decomposer_strict_fixed/T124_todo_calendar_conflict_e8ac5cec.jsonl` | 16 delegations, 16 Executor `missing_report`, 0 Executor tool calls |

No smoke trace contains protocol-correction text, transitional retry text,
synthetic failure report text, or `synthetic_failure` report statuses. These
smokes were run with `--skip-grade`; they validate protocol/config behavior,
not task quality.

## Full No-Protocol Fairness Run

Date: 2026-06-26.

Ran the full no-protocol comparison on tasks `T112,T116,T118,T124,T114,T120,T126,T128`:

- Flat Qwen3.5 Executors: 0.8B, 2B, 4B, 9B.
- Decomposer: Qwen3.6-27B manager with Qwen3.5 Executors 0.8B, 2B, 4B, 9B.
- Decomposer variants: manager strict tool-name guidance on and off.
- Flat caps kept for comparison: `react.max_turns=352`,
  `react.max_environment_tool_calls=320`.
- Decomposer budget: 16 delegations, 32 manager turns, Executor max 20 turns
  and 20 environment tool calls per delegation.
- Disabled flat/Executor ReAct repairs:
  `retry_empty_model_response=false`,
  `retry_missing_required_tool=false`,
  `retry_transitional_tool_text=false`.
- Kept manager control-loop corrections enabled.

Generation traces use tag `fair_no_protocol_26-06-26`. The strict 9B
decomposer generation had to be relaunched with
`--vllm-gpu-memory-utilization 0.86` after unrelated GPU jobs caused Qwen3.6-27B
startup OOMs. Grading used Qwen3.6-27B as judge; because unrelated short-lived
GPU jobs also interfered with judge startup, the final successful judge server
used `--vllm-max-model-len 32768`, `--vllm-gpu-memory-utilization 0.80`, and
`--enforce-eager`.

### Aggregate Results

| Variant | 0.8B | 2B | 4B | 9B |
|---|---:|---:|---:|---:|
| flat | 0.20875 / 0 pass | 0.20000 / 0 pass | 0.20000 / 0 pass | 0.20875 / 0 pass |
| decomposer no strict tools | 0.20300 / 0 pass | 0.20300 / 0 pass | 0.20630 / 0 pass | 0.20075 / 0 pass |
| decomposer strict tools | 0.20000 / 0 pass | 0.20000 / 0 pass | 0.20000 / 0 pass | 0.20000 / 0 pass |

### Per-Task Scores

| Variant | T112 | T116 | T118 | T124 | T114 | T120 | T126 | T128 | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat 0.8B | 0.20 | 0.20 | 0.20 | 0.27 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20875 |
| flat 2B | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20000 |
| flat 4B | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20000 |
| flat 9B | 0.20 | 0.20 | 0.20 | 0.27 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20875 |
| no-strict 0.8B | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.22 | 0.20 | 0.20 | 0.20300 |
| no-strict 2B | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.22 | 0.20 | 0.20 | 0.20300 |
| no-strict 4B | 0.20 | 0.20 | 0.20 | 0.20 | 0.25 | 0.20 | 0.20 | 0.20 | 0.20630 |
| no-strict 9B | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.21 | 0.20 | 0.20 | 0.20075 |
| strict 0.8B | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20000 |
| strict 2B | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20000 |
| strict 4B | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20000 |
| strict 9B | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20000 |

### Trace Diagnostics

No trace contains the disabled flat/Executor ReAct repair strings:
empty-model-response correction, missing-required-tool correction,
transitional-tool-text correction, or synthetic failure reports.

Tool-use and Executor-report diagnostics:

| Variant | Env tool calls | Delegations | Executor env tool calls | Missing reports | Nonempty reports |
|---|---:|---:|---:|---:|---:|
| flat 0.8B | 0 | 0 | 0 | 0 | 0 |
| flat 2B | 0 | 0 | 0 | 0 | 0 |
| flat 4B | 0 | 0 | 0 | 0 | 0 |
| flat 9B | 0 | 0 | 0 | 0 | 0 |
| no-strict 0.8B | 0 | 83 | 0 | 83 | 0 |
| no-strict 2B | 0 | 76 | 0 | 74 | 2 |
| no-strict 4B | 0 | 72 | 0 | 35 | 37 |
| no-strict 9B | 0 | 83 | 0 | 83 | 0 |
| strict 0.8B | 0 | 128 | 0 | 128 | 0 |
| strict 2B | 0 | 121 | 0 | 121 | 0 |
| strict 4B | 0 | 108 | 0 | 72 | 36 |
| strict 9B | 0 | 120 | 0 | 120 | 0 |

Interpretation:

1. With all compatibility retries disabled, the flat baseline also fails to use
   environment tools. This means the earlier strong flat/decomposer results
   depended heavily on the transitional-tool-text retry and related protocol
   assistance.
2. The fair no-protocol comparison collapses to the rubric floor: most tasks
   get 0.20 from non-task dimensions while outcome components are zero.
3. Strict manager tool-name guidance improves subtask wording but does not make
   the Executor issue native tool calls. It can reduce incidental nonempty text
   reports and therefore often scores exactly at the floor.
4. No-strict 4B sometimes emits nonempty Executor text reports and gets tiny
   partial credit, but those reports are not based on environment tool calls.
5. The current bottleneck is not only decomposer planning. It is ReAct/native
   tool-call adherence under vanilla no-repair serving. The next comparison
   should explicitly separate original Claw-Eval behavior from local vLLM
   compatibility repairs, because removing those repairs changes both flat and
   decomposer into mostly no-tool systems.
