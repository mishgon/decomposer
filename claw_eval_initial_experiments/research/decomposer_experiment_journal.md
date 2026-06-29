# Decomposer Experiment Journal

## 2026-06-02 - Strict executor reporting smoke on T112

### Setup

- Task: `T112_expense_email_check`.
- Mode: `batch-decomposer`, `--skip-grade`, `--no-sandbox`.
- Manager: `Qwen/Qwen3.6-27B` served by local vLLM on GPU 0.
- Executors: `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B` served by local vLLM on GPU 1.
- vLLM tool setup: `--enable-auto-tool-choice`, `--tool-call-parser hermes`.
- Manager reasoning parser: `--reasoning-parser deepseek_r1`.
- Executor reasoning parser: disabled; executor config sends `chat_template_kwargs.enable_thinking=false`.
- Report mode: `decomposer.executor_report_mode: strict`.
- Judge: disabled for this smoke run.

Strict mode means the manager only receives natural executor reports or synthetic executor failure reports. The old no-tools report repair prompt is disabled.

### Results

| Executor | Trace | Delegations | Report statuses | Outcome |
|---|---|---:|---|---|
| `Qwen3.5-2B` | `traces/Qwen_Qwen3.6-27B__Qwen_Qwen3.5-2B_26-06-02-13-35/T112_expense_email_check_abec1bc0.jsonl` | 3 started, 2 completed | 0 natural, 2 synthetic failure | Run crashed during delegation 3 with vLLM/OpenAI client error: integer string conversion exceeded 4300 digits. |
| `Qwen3.5-4B` | `traces/Qwen_Qwen3.6-27B__Qwen_Qwen3.5-4B_26-06-02-13-40/T112_expense_email_check_e029941c.jsonl` | 5 | 0 natural, 5 synthetic failure | Executor used tools but never produced valid natural reports; manager concluded tools were unavailable and submitted an incorrect unable-to-complete final answer. |
| `Qwen3.5-9B` | `traces/Qwen_Qwen3.6-27B__Qwen_Qwen3.5-9B_26-06-02-13-45/T112_expense_email_check_47103657.jsonl` | 3 | 2 natural, 1 synthetic failure | Executor produced useful natural reports for email extraction and finance lookup. Manager then delegated final report synthesis unnecessarily; final `submit_final_answer` had empty tool input, so trace ended with `no_final_answer`. |

### Observations

- Strict mode exposes the report-generation bottleneck that repair mode previously hid.
- All tested executors can emit native tool calls with the vLLM/Hermes setup, but natural reporting is model-size sensitive.
- `2B` and `4B` repeatedly produced empty visible responses after tool results. Under strict mode these become synthetic failure reports, which is aligned with the supervisor abstraction.
- `9B` is the first viable strict executor on this task: it produced natural reports for the two evidence-gathering subtasks.
- The manager currently misreads synthetic failure reports. The `stopped_reason=no_tools` field means the ReAct loop ended with no tool call in the final assistant turn, not that tools were unavailable. In the `4B` run this caused a wrong final answer claiming the executor had no tools.
- The `9B` run exposed a coordinator protocol bug: empty `submit_final_answer` input is accepted as a final-answer action, then the trace ends with `no_final_answer`.

### Current Conclusion

The strict experiment is now measuring the intended abstraction: manager sees reports or failures, not raw tool outputs. The immediate result is negative for `2B` and `4B` as strict executors on T112, and partially positive for `9B`. The bottleneck is no longer native tool calling; it is the executor's natural transition from tool-use loop to compact report, plus manager handling of failure reports.

### Next Fixes Before Another Strict Run

1. Change manager-facing synthetic reports to avoid the ambiguous phrase `stopped_reason=no_tools`; use a clearer label such as `executor_finished_without_report`.
2. Reject empty `submit_final_answer` tool inputs and issue a decomposer protocol correction instead of ending with `no_final_answer`.
3. Consider increasing `decomposer_max_output_tokens` from 1024 to 2048, because the 9B run's manager reasoning suggests it knew the final answer but did not serialize it into the tool input.
4. Rerun strict decomposer on T112 with `4B` and `9B` after those coordinator-side fixes.
5. Keep repair mode only as a debugging/completion baseline, not as the main supervisor-aligned measurement.

## 2026-06-02 - Coordinator-side strict-mode fixes implemented

Implemented fixes after the first strict T112 run:

- Manager-facing synthetic executor failures now say `executor_finished_without_report` and explicitly warn that this is not evidence that tools or data are unavailable. Raw `stopped_reason` remains trace metadata only.
- Empty `submit_final_answer` calls are rejected with a decomposer protocol error, so a manager can recover instead of ending with `no_final_answer`.
- `decomposer_max_output_tokens` was raised from 1024 to 2048 to give the manager enough room to serialize final answers into the native control tool input.
- The decomposer prompt now emphasizes smaller, single-source subtasks for SLM executors and tells the manager not to delegate final report synthesis when it already has enough evidence.
- Focused tests passed: `pytest tests/test_decomposer.py tests/test_react_loop.py tests/test_decomposer_prompts.py` (23 tests).

Next run should repeat strict T112 for `4B` and `9B`, then add microtask diagnostics for `0.8B` and `2B`.


## 2026-06-02 - Post-fix T112 reruns and SLM microdiagnostics

### Additional Hardening

- Local vLLM subprocesses now launch with `PYTHONINTMAXSTRDIGITS=0` so server-side Hermes partial JSON parsing does not crash on huge malformed numeric literals.
- The OpenAI-compatible provider now catches `ValueError` while parsing native tool arguments and falls back to empty args instead of crashing the run.
- Executor ReAct turns now use `decomposer.executor_max_output_tokens: 1024` to prevent tiny models from over-generating malformed required-tool payloads.
- Focused tests passed: `pytest tests/test_decomposer.py tests/test_react_loop.py tests/test_vllm_config.py tests/test_openai_compat.py` (30 tests).

### Rerun Results

| Executor | Task | Trace | Delegations | Report statuses | Outcome |
|---|---|---|---:|---|---|
| `Qwen3.5-4B` | full `T112_expense_email_check` | `traces/Qwen_Qwen3.6-27B__Qwen_Qwen3.5-4B_26-06-02-14-26/T112_expense_email_check_e635d7ed.jsonl` | 5 | 0 natural, 5 synthetic failure | Tool calls work, but executor never produced a valid report; manager ended with an unable-to-complete answer. |
| `Qwen3.5-9B` | full `T112_expense_email_check` | `traces/Qwen_Qwen3.6-27B__Qwen_Qwen3.5-9B_26-06-02-14-30/T112_expense_email_check_05f28e7a.jsonl` | 2 | 2 natural | Successful trace-only behavior: email report, finance report, and final verification report with INV-002 discrepancy and INV-003 pending status. |
| `Qwen3.5-2B` | `/tmp/T112_email_micro` | `traces/Qwen_Qwen3.6-27B__Qwen_Qwen3.5-2B_26-06-02-14-35/T112_email_micro_bae72a47.jsonl` | 8 | 0 natural, 8 synthetic failure | Even on a single-source email diagnostic, executor called tools but never returned a valid report. |
| `Qwen3.5-0.8B` | `/tmp/T112_email_micro` | `traces/Qwen_Qwen3.6-27B__Qwen_Qwen3.5-0.8B_26-06-02-14-51/T112_email_micro_61624bb0.jsonl` | 2 | 2 natural | Infrastructure is now stable for 0.8B; the model produced reports, but failed to fetch message bodies and therefore missed invoice numbers. |

### Interpretation

The full-task proof of concept is positive for a `9B` executor: the manager+executor abstraction works when executor reports are reliable. For smaller SLMs, the current bottleneck is no longer native tool calling or vLLM serving; it is instruction following after tool use, especially producing a compact report and using the next required tool instead of stopping after a list call. `0.8B` is back in scope as a lower-bound diagnostic, but it needs stronger executor scaffolding or training before it is useful on full benchmark tasks.

## 2026-06-02 - Executor thinking-on diagnostic for SLM report transition

### Setup

- Task: `/tmp/T112_email_micro/task.yaml`.
- Temporary config: `/tmp/config_decomposer_thinking_on_relaxed.yaml`.
- Executor `chat_template_kwargs.enable_thinking: true`.
- Relaxed executor limits: `executor_max_output_tokens: 4096`, `executor_report_max_tokens: 2048`, `executor_max_environment_tool_calls: 40`.
- Report mode remained strict; judge remained disabled.
- vLLM setup remained `--enable-auto-tool-choice --tool-call-parser hermes`; executor server still had no reasoning parser.

### Results

| Executor | Trace | Delegations | Report statuses | Outcome |
|---|---|---:|---|---|
| `Qwen3.5-2B` | `traces/thinking_on_relaxed/T112_email_micro_ad480b1a.jsonl` | 3 | 1 natural, 2 synthetic failure | Thinking-on removed empty visible responses and improved tool chaining: executor fetched message bodies and produced one valid report with invoice numbers. Two delegations still failed because report-like text was preceded by transitional phrases and exhausted the transitional-tool retry budget. |
| `Qwen3.5-4B` | `traces/thinking_on_relaxed/T112_email_micro_03773d1e.jsonl` | 5 | 0 natural, 5 synthetic failure | Executor consistently called tools and fetched message bodies, but every final assistant text was treated as transitional before report acceptance. Several messages contained useful report content after visible `</think>` markers, but strict mode rejected them because the loop stopped with `transitional_tool_retry_exhausted`. |

### Interpretation

Enabling thinking appears to help the SLMs plan multi-step tool use. The old `2B` thinking-off microtask had 0 natural reports, repeated empty visible outputs, and only `gmail_list_messages`; thinking-on produced no empty visible outputs, used `gmail_get_message`, and got one natural report. The tradeoff is that without an executor reasoning parser, Qwen thinking markers leak into visible `assistant.content`, and our transitional-text retry/validator can reject otherwise useful report content. The next fix should not simply flip the default to thinking-on. Better options are to either add an executor reasoning parser when thinking is enabled, strip/parse visible thinking markers before report validation, or make the transitional retry ignore messages that already contain report sections such as `What I did` / `Key findings`.

Follow-up default change: hierarchical runs now keep the first-tool requirement with `decomposer.executor_min_tool_calls: 1` but disable `react.retry_transitional_tool_text` in `config_decomposer.yaml`. This keeps the executor aligned with the manager/subagent protocol while avoiding forced extra tool calls when the executor already produced report-like final text.

## 2026-06-02 - Executor reasoning parser enabled diagnostic

### Setup

- Code change: executor vLLM servers now receive `--reasoning-parser deepseek_r1` by default, same as manager/model servers.
- Task/config: same `/tmp/T112_email_micro/task.yaml` and `/tmp/config_decomposer_thinking_on_relaxed.yaml` from the thinking-on diagnostic.
- Executor thinking remained enabled in request body; report mode remained strict.

### Results

| Executor | Trace | Delegations | Report statuses | Outcome |
|---|---|---:|---|---|
| `Qwen3.5-2B` | `traces/thinking_on_with_executor_parser/T112_email_micro_a6f1e3e3.jsonl` | 8 | 2 natural, 6 synthetic failure | Parser removed visible `</think>` leakage and populated `reasoning_content`, but 6 final assistant turns had output tokens with whitespace-only visible content. The decomposer hit timeout after repeated failures. |
| `Qwen3.5-4B` | `traces/thinking_on_with_executor_parser/T112_email_micro_ab3e2188.jsonl` | 8 | 1 natural, 7 synthetic failure | Parser removed visible thinking markers, but the executor mostly stopped after `gmail_list_messages`; 7 final assistant turns had whitespace-only visible content and no `gmail_get_message` calls were made. |

### Interpretation

Adding the executor reasoning parser is mechanically correct and keeps thinking text out of visible reports, but it is not sufficient for SLM executor reporting. With thinking enabled, the useful continuation often lands in `reasoning_content` while `assistant.content` is blank or whitespace, so strict report validation still fails. Compared with parser-off thinking-on, parser-on reduces visible `</think>` leakage but can reduce visible report availability and tool chaining. The next implementation fix should preserve the parser but add an explicit report scaffold or fallback extraction policy, such as a no-tools report phase that asks for visible `assistant.content`, or a structured `submit_report` control tool for the executor.

## 2026-06-26 - Transitional retry disabled for runnable configs

The transitional tool-text retry is now disabled in the flat local vLLM config
and every decomposer experiment config. Existing traces and tables that mention
retry remain historical records, but future flat/decomposer runs should compare
without this protocol repair. The ReAct implementation remains available only
as a dormant compatibility option for old configs and targeted unit tests.
