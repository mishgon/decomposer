<div align="center">

<img src="claw_eval.png" width="160" alt="Claw-Eval Logo">

# Claw-Eval

[![Tasks](https://img.shields.io/badge/tasks-300-blue)](#tasks)
[![Models](https://img.shields.io/badge/models-14-green)](#leaderboard)
[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2604.06132v1)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-purple)](https://claw-eval.github.io)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/claw-eval/Claw-Eval)
[![Dataset](https://img.shields.io/badge/ModelScope-MyRepo-624aff?logo=modelscope)](https://modelscope.cn/datasets/claw-eval/Claw-Eval)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

> Claw-Eval: Towards Trustworthy Evaluation of Autonomous Agents. <br>
> 300 human-verified tasks | 2,159 rubrics | 9 categories | Completion · Safety · Robustness.

</div>


---

## Leaderboard

Browse the full leaderboard and individual task cases at **[claw-eval.github.io](https://claw-eval.github.io)**.

**Evaluation Logic (Updated March 2026):**

* **Primary Metric: Pass^3.** To eliminate "lucky runs," a model must now consistently pass a task across **three independent trials** ($N=3$) to earn a success credit.
* **Strict Pass Criterion:** Under the Pass^3 methodology, a task is only marked as passed if the model meets the success criteria in **all three runs**.
* **Reproducibility:** We are committed to end-to-end reproducibility. Our codebase is currently being audited to ensure **all benchmark results on the leaderboard can be verified by the community**.
* **Handling API Instability**: In the event of execution errors caused by network or API fluctuations, we manually re-trigger the evaluation to ensure exactly **3** trajectories are successfully generated.

## Get Involved

We sincerely thank the teams behind [Meta (Muse Spark)](https://x.com/alexandr_wang/status/2045348588734066794?s=20), [KAT-Coder-V2](https://arxiv.org/abs/2603.27703), [Kimi](https://www.kimi.com/blog/kimi-k2-6), [Qwen](https://qwen.ai/blog?id=qwen3.6), [Tencent Hunyuan](https://github.com/Tencent-Hunyuan/Hy3-preview), [Xiaomi MiMo](https://mimo.xiaomi.com/mimo-v2-5-pro), [Z.AI / GLM](https://docs.z.ai/guides/vlm/glm-5v-turbo#pure-text-coding-tasks) and [Ant Ling](https://x.com/AntLingAGI/status/2046661013639209113) for publicly referencing, evaluating on, and engaging with Claw-Eval. We are grateful for this recognition, and we hope Claw-Eval can help the community jointly build a more scientific foundation for evaluating the general agentic capabilities of foundation models.

To run Claw-Eval and submit results to join the leaderboard, contact: **bwye@stu.pku.edu.cn**, **lirang410@gmail.com**, **nlp.lilei@gmail.com**.

## 📢 Updates
* **v1.1.0** — 300 human-verified tasks in 9 categories: Agents perceive, reason, create, and deliver.

* **v1.0.0** — Built on reproducible real-world complexity.
* **v0.0.0** — From chatbot to real world. (2026.3)



## Tasks

300 tasks across 3 splits and 9 categories, each task with human-verified rubrics.

| Split | Count | Description |
|-------|-------|-------------|
| `general` | 161 | Core agent tasks across communication, finance, ops, productivity, etc. |
| `multimodal` | 101 | Perception and creation — webpage generation, video QA, document extraction, etc. |
| `multi_turn` | 38 | Conversational tasks with simulated user personas for clarification and advice |

Agents are graded on three dimensions through full-trajectory auditing:
- **Completion** — did the agent finish the task?
- **Safety** — did it avoid harmful or unauthorized actions?
- **Robustness** — does it pass consistently across multiple trials?

### Dataset

Available on Hugging Face: [claw-eval/Claw-Eval](https://huggingface.co/datasets/claw-eval/Claw-Eval)

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Unique task identifier |
| `query` | string | Task instruction / description |
| `fixture` | list[string] | Fixture files required (available in `data/fixtures.tar.gz`) |
| `language` | string | `en` or `zh` |
| `category` | string | Task domain |

---

## Quick Start

We recommend using [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management:

```bash
pip install uv
uv venv --python 3.11
source .venv/bin/activate
```

Prepare your keys and set up the environments with one command:

```bash
export OPENROUTER_API_KEY=sk-or-...
export SERP_DEV_KEY=... # add this for tasks need real web search.  You can get api key from https://www.novada.com for convenience.
bash scripts/test_sandbox.sh
```

> **Note on video fixtures:** Due to file size limits, this GitHub repository does not include video files for video-related tasks. The complete fixtures (including all videos) are available on Hugging Face: [claw-eval/Claw-Eval](https://huggingface.co/datasets/claw-eval/Claw-Eval).

> **Note on grade:** we use **gemini-3-flash** in general and multimodal tasks while **claude opus4.6** for both grader and user-agent in multi_turn tasks!

Go rock 🚀

```bash
claw-eval batch --config model_configs/claude_opus_46.yaml --sandbox --trials 3 --parallel 16
# For different tasks, you can follow different config: config_general.yaml/config_multimodal.yaml/config_user_agent.yaml.
```

### Local vLLM Qwen3/XML Pipeline

This section describes the local pipeline used for the Qwen3.5/Qwen3.6 flat
and decomposer experiments. It is separate from the public leaderboard setup:
the goal is trace-level research on manager/executor coordination, not a
leaderboard submission.

Use the experiment environment that has vLLM and the decomposer dependencies:

```bash
conda activate sukhorukov_decomposer
# or use the environment Python directly:
export PY=/home/jovyan/.mlspace/envs/sukhorukov_decomposer/bin/python
```

The Qwen3 local experiments must use the Qwen XML native-tool parser:

```bash
--vllm-extra-arg "--tool-call-parser qwen3_xml"
```

This is important. Without the correct parser, Qwen models may emit tool-like
text that vLLM does not convert into native tool calls, and flat baselines can
appear to use no tools. The launcher also enables vLLM auto tool choice and
uses the local `deepseek_r1` reasoning parser. `deepseek_r1` is a vLLM parser
name, not an external API dependency.

The current research comparison disables generation protocol retries:

```yaml
react:
  retry_empty_model_response: false
  retry_missing_required_tool: false
  retry_transitional_tool_text: false
```

The main 8-task set used in the decomposer experiments is:

```bash
export TASKS=T112,T114,T116,T118,T120,T124,T126,T128
```

These are medium workflow/ops tasks with mock services only, so the local runs
use `--no-sandbox`.

#### 1. Smoke Test

Run one small decomposer smoke test before launching a matrix:

```bash
$PY -m claw_eval.cli batch-decomposer \
  --config configs/experiments/config_decomposer_synth_report_wrapper_no_strict_budget16_min0.yaml \
  --tasks T112 \
  --decomposer-model Qwen/Qwen3.6-27B \
  --executor-model Qwen/Qwen3.5-0.8B \
  --trials 1 \
  --parallel 1 \
  --no-sandbox \
  --skip-grade \
  --trace-dir traces/smoke_decomposer_qwen3xml \
  --launch-vllm \
  --stop-vllm-on-exit \
  --decomposer-gpu 0 \
  --decomposer-port 8000 \
  --executor-gpu 1 \
  --executor-port 8001 \
  --vllm-max-model-len 65536 \
  --vllm-gpu-memory-utilization 0.92 \
  --vllm-extra-arg "--tool-call-parser qwen3_xml"
```

Check the vLLM logs:

```bash
rg "tool_call_parser|qwen3_xml|auto.*tool choice|reasoning_parser" logs/vllm/*.log
```

Healthy logs should show `tool_call_parser='qwen3_xml'` and `"auto" tool
choice enabled.

#### 2. Flat Baseline

The flat baseline runs one model directly on the original task. To make it
comparable to decomposer runs with 16 delegations and 20 executor tool calls per
delegation, use the flat cap surface in `configs/experiments/config_vllm_judge_no_thinking.yaml`:

```yaml
react:
  max_turns: 352
  max_environment_tool_calls: 320
```

Run a flat baseline for one model:

```bash
MODEL=Qwen/Qwen3.5-9B
TAG=flat_9b_qwen3xml_no_retry

$PY -m claw_eval.cli batch \
  --config configs/experiments/config_vllm_judge_no_thinking.yaml \
  --tasks "$TASKS" \
  --model "$MODEL" \
  --trials 1 \
  --parallel 1 \
  --no-sandbox \
  --skip-grade \
  --trace-dir "traces/$TAG" \
  --launch-vllm \
  --stop-vllm-on-exit \
  --model-gpu 0 \
  --model-port 8000 \
  --vllm-max-model-len 65536 \
  --vllm-gpu-memory-utilization 0.92 \
  --vllm-extra-arg "--tool-call-parser qwen3_xml"
```

For a full flat matrix, loop over executor sizes:

```bash
for spec in \
  "0_8b:Qwen/Qwen3.5-0.8B" \
  "2b:Qwen/Qwen3.5-2B" \
  "4b:Qwen/Qwen3.5-4B" \
  "9b:Qwen/Qwen3.5-9B"
do
  key=${spec%%:*}
  model=${spec#*:}
  $PY -m claw_eval.cli batch \
    --config configs/experiments/config_vllm_judge_no_thinking.yaml \
    --tasks "$TASKS" \
    --model "$model" \
    --trials 1 \
    --parallel 1 \
    --no-sandbox \
    --skip-grade \
    --trace-dir "traces/qwen3xml_flat_no_retry_${key}" \
    --launch-vllm \
    --stop-vllm-on-exit \
    --model-gpu 0 \
    --model-port 8000 \
    --vllm-max-model-len 65536 \
    --vllm-gpu-memory-utilization 0.92 \
    --vllm-extra-arg "--tool-call-parser qwen3_xml"
done
```

#### 3. Decomposer Runs

The decomposer setup runs a manager model and an executor model:

- manager/decomposer: `Qwen/Qwen3.6-27B` on GPU 0, port 8000
- executor: Qwen3.5 size under test on GPU 1, port 8001
- decomposer budget: 32 manager turns, 16 delegations
- executor budget: 20 turns, 20 environment tool calls
- executor minimum tool calls: 0
- thinking: disabled for manager, executor, and judge through config
- protocol retries: disabled

Important decomposer configs:

| config | executor prompt | strict tool guidance | synthetic failure report |
|---|---|---:|---:|
| `config_decomposer_no_retry_flat_subtask_no_strict_budget16_min0.yaml` | flat subtask | off | off |
| `config_decomposer_no_retry_flat_subtask_strict_tools_budget16_min0.yaml` | flat subtask | on | off |
| `config_decomposer_no_retry_report_wrapper_no_strict_budget16_min0.yaml` | report wrapper | off | off |
| `config_decomposer_no_retry_report_wrapper_strict_tools_budget16_min0.yaml` | report wrapper | on | off |
| `config_decomposer_synth_report_wrapper_no_strict_budget16_min0.yaml` | report wrapper | off | on |
| `config_decomposer_synth_report_wrapper_strict_tools_budget16_min0.yaml` | report wrapper | on | on |

`manager_valid_tool_guidance: true` means the manager prompt includes valid tool
names for the current task and is encouraged to mention those names in
delegated subtasks. If it is `false`, the manager still delegates through the
same `delegate_subtask` control tool, but it does not receive that explicit
tool-name guidance.

`executor_synthetic_failure_report: true` changes only the manager/executor
communication when the executor uses tools but fails to produce a visible
report. The manager receives an explicit report such as
`executor_failed_no_report` instead of an empty result. It does not expose raw
executor tool evidence to the manager.

Run one decomposer mode for one executor:

```bash
CFG=configs/experiments/config_decomposer_synth_report_wrapper_no_strict_budget16_min0.yaml
EXECUTOR=Qwen/Qwen3.5-4B
TAG=decomposer_synth_report_no_strict_4b

$PY -m claw_eval.cli batch-decomposer \
  --config "$CFG" \
  --tasks "$TASKS" \
  --decomposer-model Qwen/Qwen3.6-27B \
  --executor-model "$EXECUTOR" \
  --trials 1 \
  --parallel 1 \
  --no-sandbox \
  --skip-grade \
  --trace-dir "traces/$TAG" \
  --launch-vllm \
  --stop-vllm-on-exit \
  --decomposer-gpu 0 \
  --decomposer-port 8000 \
  --executor-gpu 1 \
  --executor-port 8001 \
  --vllm-max-model-len 65536 \
  --vllm-gpu-memory-utilization 0.92 \
  --vllm-extra-arg "--tool-call-parser qwen3_xml"
```

Run the synthetic-failure report ablation matrix:

```bash
export LOGDIR=logs/experiments/qwen3xml_synth_failure_$(date +%d-%m-%H)
mkdir -p "$LOGDIR"
: > "$LOGDIR/trace_dirs.txt"

models=(
  "0_8b:Qwen/Qwen3.5-0.8B"
  "2b:Qwen/Qwen3.5-2B"
  "4b:Qwen/Qwen3.5-4B"
  "9b:Qwen/Qwen3.5-9B"
)
modes=(
  "decomposer_report_no_strict:configs/experiments/config_decomposer_synth_report_wrapper_no_strict_budget16_min0.yaml"
  "decomposer_report_strict:configs/experiments/config_decomposer_synth_report_wrapper_strict_tools_budget16_min0.yaml"
)

for model_spec in "${models[@]}"; do
  model_key=${model_spec%%:*}
  executor_model=${model_spec#*:}
  for mode_spec in "${modes[@]}"; do
    mode=${mode_spec%%:*}
    cfg=${mode_spec#*:}
    root="traces/qwen3xml_synth_failure_${mode}_${model_key}"
    echo "$root" >> "$LOGDIR/trace_dirs.txt"

    $PY -m claw_eval.cli batch-decomposer \
      --config "$cfg" \
      --tasks "$TASKS" \
      --decomposer-model Qwen/Qwen3.6-27B \
      --executor-model "$executor_model" \
      --trials 1 \
      --parallel 1 \
      --no-sandbox \
      --skip-grade \
      --trace-dir "$root" \
      --launch-vllm \
      --stop-vllm-on-exit \
      --decomposer-gpu 0 \
      --decomposer-port 8000 \
      --executor-gpu 1 \
      --executor-port 8001 \
      --vllm-max-model-len 65536 \
      --vllm-gpu-memory-utilization 0.92 \
      --vllm-extra-arg "--tool-call-parser qwen3_xml" \
      > "$LOGDIR/${mode}_${model_key}.log" 2>&1
  done
done
```

Each decomposer `--trace-dir` contains one nested timestamped batch directory,
for example:

```text
traces/qwen3xml_synth_failure_decomposer_report_no_strict_4b/
  Qwen_Qwen3.6-27B__Qwen_Qwen3.5-4B_<timestamp>/
    batch_results.json
    T112_...jsonl
    T112_..._exec_1.jsonl
```

Build a manifest of the nested batch directories:

```bash
python - <<'PY'
import os
from pathlib import Path
logdir = Path(os.environ["LOGDIR"])
outer = logdir / "trace_dirs.txt"
out = logdir / "batch_dirs.txt"
rows = []
for line in outer.read_text().splitlines():
    root = Path(line)
    rows.extend(str(p.parent) for p in sorted(root.glob("*/batch_results.json")))
out.write_text("\n".join(rows) + "\n")
print(out)
PY
```

Adjust `logdir` to your experiment tag.

#### 4. Grading Existing Traces

Grade with the local judge config:

```bash
JUDGE_CFG=configs/experiments/config_vllm_judge_no_thinking.yaml
JUDGE_MODEL=Qwen/Qwen3.6-27B
```

Grade one batch directory:

```bash
$PY -m claw_eval.cli grade-batch \
  --trace-dir "$BATCH_DIR" \
  --tasks-dir tasks \
  --config "$JUDGE_CFG" \
  --judge-model "$JUDGE_MODEL" \
  --launch-vllm \
  --judge-gpu 0 \
  --judge-port 8000 \
  --vllm-max-model-len 65536 \
  --vllm-gpu-memory-utilization 0.92 \
  --vllm-extra-arg "--tool-call-parser qwen3_xml" \
  --force
```

Grade all decomposer batch dirs from a manifest:

```bash
while IFS= read -r dir; do
  [ -n "$dir" ] || continue
  echo "[GRADE] $dir"
  $PY -m claw_eval.cli grade-batch \
    --trace-dir "$dir" \
    --tasks-dir tasks \
    --config "$JUDGE_CFG" \
    --judge-model "$JUDGE_MODEL" \
    --launch-vllm \
    --judge-gpu 0 \
    --judge-port 8000 \
    --vllm-max-model-len 65536 \
    --vllm-gpu-memory-utilization 0.92 \
    --vllm-extra-arg "--tool-call-parser qwen3_xml" \
    --force
done < "$LOGDIR/batch_dirs.txt"
```

`--force` appends a fresh grading result to each trace and updates
`batch_results.json`. Use it when comparing runs under a new judge or fixed
grading logic.

#### 5. Aggregating Results

Every graded batch directory has a `batch_results.json`. A quick aggregate:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

for d in (Path(os.environ["LOGDIR"]) / "batch_dirs.txt").read_text().splitlines():
    p = Path(d) / "batch_results.json"
    data = json.loads(p.read_text())
    scores = [row["trials"][-1]["task_score"] for row in data]
    passes = [row["trials"][-1]["passed"] for row in data]
    print(f"{p.parent}: avg={sum(scores)/len(scores):.3f} pass={sum(passes)}/{len(passes)}")
PY
```

For the latest synthetic-failure run, the full comparison table is stored at:

```text
logs/experiments/qwen3xml_synth_failure_29-06-29/synthetic_failure_comparison.md
```

Trace examples for lab meetings are summarized in:

```text
research/decomposer_manager_executor_examples.md
```

#### 6. Interpreting Decomposer Traces

Manager traces contain:

- the original task message fed to the manager
- `delegation_start` events with manager-created subtasks
- `delegation_end` events with executor reports
- `decomposer_summary` with counts such as `executor_report_status_counts`

Executor sidecar traces are named with `_exec_<n>.jsonl` and contain the
executor's ReAct loop for one delegated subtask.

Useful fields in `delegation_end`:

| field | meaning |
|---|---|
| `report_status: natural` | executor produced visible report text naturally |
| `report_status: synthetic_failure` | executor used tools but failed to produce a valid report |
| `executor_stopped_reason: no_tools` | executor stopped by returning text/no tool calls |
| `executor_stopped_reason: max_turns` | executor hit the turn cap |
| `executor_stopped_reason: tool_budget` | executor hit the environment tool-call cap |
| `executor_environment_tool_count` | number of environment tool calls in that subtask |
| `executor_empty_visible_response_count` | empty visible model responses in the subtask |

Typical diagnostics:

```bash
rg '"report_status":"synthetic_failure"|"executor_stopped_reason":"max_turns"|"executor_stopped_reason":"tool_budget"' traces/<run>/**/*.jsonl
```

The main failure mode found in the Qwen3.5 small-executor experiments is not
only blank report transport. Many weak executors repeatedly call tools until
`max_turns` or `tool_budget` and never transition to a visible report. Synthetic
failure reports make this visible to the manager, but they do not recover raw
tool evidence from the failed executor sidecar.

---

## Roadmap

- [x] More real-world, multimodal tasks in complex productivity environments
- [x] Comprehensive, fine-grained scoring logic with deep state verification
- [x] Enhanced sandbox isolation and full-trace tracking for transparent, scalable evaluation


## Contribution
We welcome any kind of contribution. Let us know if you have any suggestions!

## Acknowledgements
Our test cases are built on the work of the community. We draw from and adapt tasks contributed by OpenClaw, PinchBench, OfficeQA, OneMillion-Bench, Finance Agent, and Terminal-Bench 2.0.

## Core Contributors
[Bowen Ye](https://github.com/pkuYmiracle)(PKU), [Rang Li](https://github.com/lirang04) (PKU), [Qibin Yang](https://github.com/yangqibin-caibi) (PKU), [Zhihui Xie](https://zhxie.site/)(HKU), [Yuanxin Liu](https://llyx97.github.io/)(PKU), [Linli Yao](https://yaolinli.github.io/)(PKU), [Hanglong Lyu](https://github.com/Albus2002)(PKU), [Lei Li](lilei-nlp.github.io)(HKU, project lead)


## Advisors
[Tong Yang](https://yangtonghome.github.io/) (PKU), [Zhifang Sui](https://cs.pku.edu.cn/info/1226/2014.htm) (PKU), [Lingpeng Kong](https://ikekonglp.github.io/) (HKU), [Qi Liu](https://leuchine.github.io/) (HKU)

## Citation

If you use Claw-Eval in your research, please cite:

```bibtex
@misc{ye2026clawevaltrustworthyevaluationautonomous,
      title={Claw-Eval: Towards Trustworthy Evaluation of Autonomous Agents}, 
      author={Bowen Ye and Rang Li and Qibin Yang and Yuanxin Liu and Linli Yao and Hanglong Lv and Zhihui Xie and Chenxin An and Lei Li and Lingpeng Kong and Qi Liu and Zhifang Sui and Tong Yang},
      year={2026},
      eprint={2604.06132},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.06132}, 
}
```

## License

This project is released under the [MIT License](LICENSE).
