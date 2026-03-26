from __future__ import annotations

from ast import literal_eval
import json
from pathlib import Path


def write_dataset_reports(output_dir: str | Path) -> None:
    root = Path(output_dir)
    debug_samples_path = root / "debug_samples.json"
    if not debug_samples_path.exists():
        raise FileNotFoundError(debug_samples_path)

    debug_samples = json.loads(debug_samples_path.read_text(encoding="utf-8"))
    abstract_examples = _load_abstract_examples(root / "abstract_traces" / "train.jsonl")
    raw_examples = _load_raw_examples(root / "raw_traces")
    chain_examples = _select_chain_examples(raw_examples, abstract_examples)

    (root / "debug_samples_guide.md").write_text(_render_debug_samples_guide(), encoding="utf-8")
    (root / "chain_examples.md").write_text(_render_chain_examples(chain_examples), encoding="utf-8")
    (root / "ideal_reasoning_examples.md").write_text(
        _render_ideal_reasoning_examples(chain_examples),
        encoding="utf-8",
    )
    (root / "ideal_reasoning_examples.json").write_text(
        json.dumps({"examples": chain_examples}, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _load_abstract_examples(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _load_raw_examples(raw_root: Path) -> list[dict]:
    records = []
    if not raw_root.exists():
        return records
    for path in sorted(raw_root.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        episode = record["episode"]
        records.append(
            {
                "path": str(path),
                "episode_id": episode["episode_id"],
                "entry_function": episode["entry_func_id"],
                "input": episode["input"]["expr"]["display"],
                "final_output": episode["final_output"]["display"] if episode["final_output"] else None,
                "num_calls": episode["num_calls"],
                "pattern_label": episode["pattern_label"],
                "call_sequence": [
                    {
                        "call_id": call_id,
                        "depth": record["calls"][call_id]["depth"],
                        "func_id": record["calls"][call_id]["func_id"],
                        "output": record["calls"][call_id]["output"]["display"]
                        if record["calls"][call_id]["output"]
                        else None,
                    }
                    for call_id in episode["call_ids"]
                ],
            }
        )
    return records


def _select_chain_examples(raw_examples: list[dict], abstract_examples: list[dict]) -> list[dict]:
    raw_by_episode = {sample["episode_id"]: sample for sample in raw_examples}

    abstract_by_episode = {}
    for sample in abstract_examples:
        abstract_by_episode[sample["metadata"]["episode_id"]] = sample

    selected: list[dict] = []
    selected.extend(_pick_simplify_examples(raw_by_episode, abstract_by_episode, limit=2))
    selected.extend(_pick_solve_examples(raw_by_episode, abstract_by_episode, limit=2))
    selected.extend(_pick_factor_examples(raw_by_episode, abstract_by_episode, limit=2))
    return selected


def _pick_simplify_examples(raw_by_episode: dict, abstract_by_episode: dict, *, limit: int) -> list[dict]:
    candidates = []
    for episode_id, abstract in abstract_by_episode.items():
        if abstract["task"]["entry_function"] != "simplify":
            continue
        tools = [step["assistant"]["tool_call"]["tool"] for step in abstract["gold_trajectory"][::2]]
        if "sympy.trig_simplify" not in tools:
            continue
        raw = raw_by_episode.get(episode_id)
        candidates.append((episode_id, raw, abstract, len(set(tools)), len(tools)))
    selected = []
    seen_inputs: set[str] = set()
    for episode_id, raw_sample, abstract_sample, _, _ in sorted(
        candidates,
        key=lambda item: (item[3], item[4], item[1]["num_calls"] if item[1] else 0),
        reverse=True,
    ):
        input_display = abstract_sample["task"]["input"]["expr"]["display"]
        if input_display in seen_inputs:
            continue
        seen_inputs.add(input_display)
        selected.append(_build_chain_example("simplify_mixed", raw_sample, abstract_sample))
        if len(selected) >= limit:
            break
    return selected


def _pick_solve_examples(raw_by_episode: dict, abstract_by_episode: dict, *, limit: int) -> list[dict]:
    candidates = []
    for episode_id, sample in abstract_by_episode.items():
        if sample["task"]["entry_function"] == "solve":
            raw = raw_by_episode.get(episode_id)
            score = raw["num_calls"] if raw else 0
            trig_bonus = 1 if "sin(" in sample["task"]["input"]["expr"]["display"] else 0
            system_bonus = 1 if sample["task"]["input"]["expr"]["display"].startswith("[") else 0
            candidates.append((episode_id, raw, sample, score, trig_bonus, system_bonus))
    selected = []
    seen_inputs: set[str] = set()
    for episode_id, raw_sample, abstract_sample, _, _, _ in sorted(
        candidates,
        key=lambda item: (item[3], item[4], item[5]),
        reverse=True,
    ):
        input_display = abstract_sample["task"]["input"]["expr"]["display"]
        if input_display in seen_inputs:
            continue
        seen_inputs.add(input_display)
        selected.append(_build_chain_example("solve_branching", raw_sample, abstract_sample))
        if len(selected) >= limit:
            break
    return selected


def _pick_factor_examples(raw_by_episode: dict, abstract_by_episode: dict, *, limit: int) -> list[dict]:
    candidates = []
    for episode_id, raw in raw_by_episode.items():
        if raw["entry_function"] != "factor":
            continue
        abstract = abstract_by_episode.get(episode_id)
        candidates.append((episode_id, raw, abstract))
    selected = []
    seen_inputs: set[str] = set()
    for episode_id, raw_sample, abstract_sample in sorted(
        candidates,
        key=lambda item: item[1]["num_calls"],
        reverse=True,
    ):
        if raw_sample["input"] in seen_inputs:
            continue
        seen_inputs.add(raw_sample["input"])
        selected.append(_build_chain_example("factor_decompose", raw_sample, abstract_sample))
        if len(selected) >= limit:
            break
    return selected


def _build_chain_example(chain_type: str, raw_sample: dict | None, abstract_sample: dict | None) -> dict:
    input_value = None
    final_value = None
    if raw_sample:
        input_value = raw_sample["input"]
        final_value = raw_sample["final_output"]
    elif abstract_sample:
        input_value = abstract_sample["task"]["input"]["expr"]["display"]
        final_value = abstract_sample["final_answer"]["display"] if abstract_sample["final_answer"] else None

    tool_steps = _sanitize_tool_steps(abstract_sample)
    ideal_tool_steps = _idealize_tool_steps(chain_type, tool_steps)
    return {
        "chain_type": chain_type,
        "episode_id": raw_sample["episode_id"] if raw_sample else abstract_sample["metadata"]["episode_id"],
        "entry_function": raw_sample["entry_function"] if raw_sample else abstract_sample["task"]["entry_function"],
        "input": input_value,
        "final_output": final_value,
        "pattern_label": raw_sample["pattern_label"] if raw_sample else abstract_sample["metadata"]["pattern_label"],
        "num_calls": raw_sample["num_calls"] if raw_sample else len(tool_steps),
        "raw_call_sequence": raw_sample["call_sequence"] if raw_sample else [],
        "tool_steps": tool_steps,
        "ideal_tool_steps": ideal_tool_steps,
    }


def _tool_reason(tool_name: str) -> str:
    reasons = {
        "sympy.simplify": "Use the top-level simplifier when no more specific internal macro-step is exposed.",
        "sympy.rational_simplify": "Combine fractions, cancel common factors, and normalize multiplicative structure.",
        "sympy.trig_simplify": "Apply trigonometric identities or deep trig normalization.",
        "sympy.factor": "Return the factorized form of the current expression.",
        "sympy.factor_decompose": "Perform internal polynomial factor decomposition before the final factor form.",
        "sympy.solve": "Start solving the whole equation or system.",
        "sympy.inspect_branches": "Inspect the solver state to expose explicit candidate solution branches.",
        "sympy.solve_branch": "Solve one branch or reduced subproblem produced by the solver.",
        "sympy.expand": "Expand products and powers into additive form.",
    }
    return reasons.get(tool_name, "Apply the next SymPy transformation indicated by the reference trajectory.")


def _sanitize_tool_steps(abstract_sample: dict | None) -> list[dict]:
    if not abstract_sample:
        return []
    task = abstract_sample["task"]["input"]
    current_handle = "$0"
    symbol_names = _parse_symbol_names(task.get("symbol_names", {}).get("display"))
    steps = []
    pending_steps = []
    current_value = task["expr"]["display"]
    for index, (assistant_step, tool_step) in enumerate(
        zip(abstract_sample["gold_trajectory"][::2], abstract_sample["gold_trajectory"][1::2]),
        start=1,
    ):
        tool_name = assistant_step["assistant"]["tool_call"]["tool"]
        clean_args = {"expr": current_handle}
        if tool_name in {"sympy.solve", "sympy.solve_branch"} and symbol_names:
            clean_args["symbol_names"] = symbol_names
        if tool_name == "sympy.inspect_branches" and symbol_names:
            clean_args["symbol_names"] = symbol_names
        tool_args = assistant_step["assistant"]["tool_call"]["args"]
        if tool_name == "sympy.solve_branch" and "branch_id" in tool_args:
            clean_args["branch_id"] = tool_args["branch_id"]
        result_handle = f"${index}"
        value = tool_step["tool"]["value"]
        pending_steps.append(
            {
                "tool": tool_name,
                "args": clean_args,
                "result": result_handle,
                "value": value,
                "reason": _tool_reason(tool_name),
            }
        )
        current_handle = result_handle
        current_value = value

    non_simplify_count = sum(1 for step in pending_steps if step["tool"] != "sympy.simplify")
    for step in pending_steps:
        if step["tool"] == "sympy.simplify" and non_simplify_count > 0:
            continue
        steps.append(step)
    return _dedupe_clean_steps(steps)


def _parse_symbol_names(display: str | None) -> list[str] | None:
    if not display:
        return None
    try:
        value = literal_eval(display)
    except Exception:
        return [display]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _dedupe_clean_steps(steps: list[dict]) -> list[dict]:
    deduped = []
    previous = None
    for idx, step in enumerate(steps, start=1):
        signature = (step["tool"], tuple(sorted(step["args"].items())), step["value"])
        if signature == previous:
            continue
        normalized = dict(step)
        normalized["result"] = f"${len(deduped) + 1}"
        normalized["args"] = {"expr": "$0" if not deduped else f"${len(deduped)}", **{k: v for k, v in step["args"].items() if k != "expr"}}
        deduped.append(normalized)
        previous = signature
    return deduped


def _idealize_tool_steps(chain_type: str, tool_steps: list[dict]) -> list[dict]:
    if chain_type == "simplify_mixed":
        kept = [step for step in tool_steps if step["tool"] in {"sympy.rational_simplify", "sympy.trig_simplify"}]
        return _dedupe_clean_steps(kept)
    if chain_type == "solve_branching":
        kept = [step for step in tool_steps if step["tool"] in {"sympy.inspect_branches", "sympy.solve_branch"}]
        return _dedupe_clean_steps(kept[:2]) if kept else tool_steps[:1]
    if chain_type == "factor_decompose":
        for step in tool_steps:
            if step["tool"] == "sympy.factor":
                return [dict(step, result="$1", args={"expr": "$0"})]
        return tool_steps[:1]
    return tool_steps


def _render_debug_samples_guide() -> str:
    return """# `debug_samples.json` Guide

`debug_samples.json` is a compact inspection artifact produced for each dataset run.

## Top-level keys

- `raw_samples`: longest raw episodes selected by raw call count
- `abstract_samples`: longest model-facing episodes selected by abstract trajectory length

## `raw_samples` fields

- `path`: source JSON file under `raw_traces/`
- `episode_id`: stable dataset episode identifier
- `entry_function`: top-level SymPy API that was run
- `input`: original serialized input expression
- `final_output`: final serialized SymPy result
- `num_calls`: number of raw traced calls
- `pattern_label`: coarse shape label such as `chain`, `loop`, `hierarchy`, or `mixed`
- `call_sequence`: flattened raw call list in execution order

## `call_sequence` fields

- `call_id`: unique call node id inside the episode
- `depth`: tree depth
- `func_id`: traced SymPy function identity
- `output`: serialized display form of the return value

## `abstract_samples` fields

- same schema as one line in `abstract_traces/train.jsonl`
- `gold_trajectory` alternates assistant tool calls and tool observations
- each pair represents the ideal model-facing step sequence for that episode
"""


def _render_chain_examples(examples: list[dict]) -> str:
    lines = ["# Chain Examples", ""]
    for example in examples:
        lines.append(f"## {example['chain_type']}")
        lines.append(f"- example_family: `{example['chain_type']}`")
        lines.append(f"- episode_id: `{example['episode_id']}`")
        lines.append(f"- entry_function: `{example['entry_function']}`")
        lines.append(f"- pattern_label: `{example['pattern_label']}`")
        lines.append(f"- input: `{example['input']}`")
        lines.append(f"- final_output: `{example['final_output']}`")
        lines.append(f"- raw_num_calls: `{example['num_calls']}`")
        lines.append("")
        lines.append("### Raw Call Prefix")
        lines.append("")
        for call in example["raw_call_sequence"][:12]:
            lines.append(f"- depth {call['depth']}: `{call['func_id']}` -> `{call['output']}`")
        lines.append("")
        lines.append("### Abstract Tool Steps")
        lines.append("")
        for step in example["tool_steps"]:
            lines.append(f"- `{step['tool']}` with args `{step['args']}` -> `{step['value']}`")
        lines.append("")
    return "\n".join(lines)


def _render_ideal_reasoning_examples(examples: list[dict]) -> str:
    lines = ["# Ideal Reasoning Examples", ""]
    for example in examples:
        lines.append(f"## {example['chain_type']}")
        lines.append("")
        lines.append(f"Task: transform `{example['input']}` to match the SymPy reference output `{example['final_output']}`.")
        lines.append("")
        for idx, step in enumerate(example["ideal_tool_steps"], start=1):
            lines.append(f"Step {idx}")
            lines.append(f"- Reasoning: {step['reason']}")
            lines.append(f"- Tool call: `{step['tool']}` with args `{step['args']}`")
            lines.append(f"- Observation: `{step['result']}` = `{step['value']}`")
            lines.append("")
        lines.append(f"Final answer: `{example['final_output']}`")
        lines.append("")
    return "\n".join(lines)
