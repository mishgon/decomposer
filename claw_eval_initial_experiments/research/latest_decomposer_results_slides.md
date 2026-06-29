<!-- Latest decomposer experiment summary, 2026-06-29 -->

# Decomposer Evaluation: Latest Controlled Runs

**Goal.** Test whether a strong manager can improve smaller tool-using executor models on Claw-Eval personal-assistant tasks.

**Experiment setup**

- Tasks: 8 general workflow tasks (`T112`, `T114`, `T116`, `T118`, `T120`, `T124`, `T126`, `T128`)
- Manager: `Qwen3.6-27B`
- Executors / flat baselines: `Qwen3.5-0.8B`, `2B`, `4B`, `9B`
- Local vLLM tool parser fixed: `qwen3_xml`
- Generation protocol retries disabled:
  - no empty-response retry
  - no missing-required-tool retry
  - no transitional-tool-text retry
- Same judge for all runs: local `Qwen3.6-27B`

**Structures tested**

| Structure | What it tests |
|---|---|
| Flat baseline | One model solves the original task directly |
| Decomposer + flat executor | Manager delegates; executor receives flat-style subtask prompt |
| Decomposer + report-wrapper executor | Manager delegates; executor is explicitly asked to finish with compact report |
| Strict manager tools | Manager sees exact valid executor tool names |
| No strict manager tools | Manager only has general delegation instructions |

```text
Flat:
Task -> Model -> Environment tools -> Final answer

Decomposer:
Task -> Manager -> delegate_subtask -> Executor -> Environment tools
     <- Executor report <- Manager -> Final answer
```

---

# Latest Results: Flat-Executor Decomposer Is Competitive

Average score with pass count in parentheses:

| mode | 0.8B | 2B | 4B | 9B |
|---|---:|---:|---:|---:|
| flat | 0.372 (0/8) | 0.626 (5/8) | 0.672 (5/8) | 0.888 (7/8) |
| decomposer flat no-strict | 0.618 (4/8) | 0.798 (6/8) | 0.816 (6/8) | 0.883 (6/8) |
| decomposer flat strict | 0.566 (3/8) | 0.841 (6/8) | 0.804 (4/8) | **0.927 (8/8)** |
| decomposer report no-strict | 0.577 (4/8) | 0.532 (3/8) | 0.590 (1/8) | 0.848 (5/8) |
| decomposer report strict | 0.765 (5/8) | **0.210 (0/8)** | 0.649 (5/8) | 0.884 (6/8) |

**Takeaways**

- Best run: **decomposer flat strict 9B** reaches `0.927`, `8/8`, above flat 9B at `0.888`, `7/8`.
- The decomposer helps smaller executors when the executor prompt matches the flat baseline style: 2B improves from `0.626` flat to `0.841` in decomposer flat strict.
- Report-wrapper strict is brittle. The 2B executor repeatedly used tools and failed to stop with a report: `110/114` delegations ended as `missing_report`.
- The old blank-report issue is now localized:
  - tool calls work after `qwen3_xml`;
  - final runs had `0` empty visible executor responses;
  - remaining failures are mostly executor non-termination before report plus manager receiving empty report text.

**Current best candidate**

Use **manager + flat executor prompt + strict manager tool guidance**.

**Next ablations**

- Pass explicit synthetic failure reports to the manager on `missing_report`.
- Test structured `submit_report` or a repair report phase.
- Track diagnostics in result tables: `missing_report`, `tool_budget`, `max_turns`, delegations, and tool calls.
