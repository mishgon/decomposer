from __future__ import annotations

from ast import literal_eval
import json
from pathlib import Path

from .sft.export_traces import _task_goal


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


def _primary_input_display(input_payload: dict) -> str | None:
    preferred = (
        "expr",
        "system",
        "inequalities",
        "basis_exprs",
        "target_expr",
        "elements",
        "sets",
        "start",
        "end",
        "left",
        "right",
        "candidate",
        "other",
        "exprs",
        "matrix_rows",
        "guess",
        "point",
        "lower",
        "upper",
        "symbol_name",
        "coeff_names",
    )
    for key in preferred:
        value = input_payload.get(key)
        if isinstance(value, dict) and "display" in value:
            return value["display"]
    for value in input_payload.values():
        if isinstance(value, dict) and "display" in value:
            return value["display"]
    return None


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
                    "input": _primary_input_display(episode["input"]),
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
    selected.extend(_pick_simplify_examples(raw_by_episode, abstract_by_episode, limit=3))
    selected.extend(_pick_solve_examples(raw_by_episode, abstract_by_episode, limit=3))
    selected.extend(_pick_factor_examples(raw_by_episode, abstract_by_episode, limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "apart", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "trigsimp", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "powsimp", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "solveset", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "factor_list", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "linsolve", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "linear_eq_to_matrix", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "groebner", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "GroebnerBasis.reduce", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "diff", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "integrate", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "limit", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "collect", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "nsimplify", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "radsimp", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "hyperexpand", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "combsimp", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "gammasimp", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "expand_mul", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "expand_trig", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "expand_log", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "expand_power_base", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "expand_power_exp", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "logcombine", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "gcd", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "gcd_list", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "lcm", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "resultant", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "terms_gcd", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "nonlinsolve", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "nsolve", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "solve_linear", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "solve_undetermined_coeffs", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "solve_linear_system", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "solve_linear_system_LU", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "solve_poly_system", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "reduce_inequalities", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "checksol", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "diophantine", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "unrad", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "Interval", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "FiniteSet", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "Union", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "Intersection", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "Complement", limit=2))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "Poly", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "residue", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "summation", limit=1))
    selected.extend(_pick_entrypoint_examples(raw_by_episode, abstract_by_episode, "simplify_logic", limit=1))
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
        input_display = _primary_input_display(abstract_sample["task"]["input"])
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
            input_display = _primary_input_display(sample["task"]["input"]) or ""
            trig_bonus = 1 if "sin(" in input_display else 0
            system_bonus = 1 if input_display.startswith("[") else 0
            candidates.append((episode_id, raw, sample, score, trig_bonus, system_bonus))
    selected = []
    seen_inputs: set[str] = set()
    for episode_id, raw_sample, abstract_sample, _, _, _ in sorted(
        candidates,
        key=lambda item: (item[3], item[4], item[5]),
        reverse=True,
    ):
        input_display = _primary_input_display(abstract_sample["task"]["input"])
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
        input_value = _primary_input_display(abstract_sample["task"]["input"])
        final_value = abstract_sample["final_answer"]["display"] if abstract_sample["final_answer"] else None

    tool_steps = _sanitize_tool_steps(abstract_sample)
    ideal_tool_steps = _idealize_tool_steps(chain_type, tool_steps)
    return {
        "chain_type": chain_type,
        "episode_id": raw_sample["episode_id"] if raw_sample else abstract_sample["metadata"]["episode_id"],
        "entry_function": raw_sample["entry_function"] if raw_sample else abstract_sample["task"]["entry_function"],
        "task_goal": abstract_sample["task"]["goal"] if abstract_sample else _task_goal(
            raw_sample["entry_function"], {"expr": {"display": input_value}}
        ),
        "input": input_value,
        "final_output": final_value,
        "pattern_label": raw_sample["pattern_label"] if raw_sample else abstract_sample["metadata"]["pattern_label"],
        "num_calls": raw_sample["num_calls"] if raw_sample else len(tool_steps),
        "raw_call_sequence": raw_sample["call_sequence"] if raw_sample else [],
        "tool_steps": tool_steps,
        "ideal_tool_steps": ideal_tool_steps,
    }


def _pick_entrypoint_examples(
    raw_by_episode: dict,
    abstract_by_episode: dict,
    entry_function: str,
    *,
    limit: int,
) -> list[dict]:
    candidates = []
    for episode_id, sample in abstract_by_episode.items():
        if sample["task"]["entry_function"] != entry_function:
            continue
        if not sample["gold_trajectory"]:
            continue
        raw = raw_by_episode.get(episode_id)
        score = raw["num_calls"] if raw else 0
        candidates.append((episode_id, raw, sample, score))

    selected = []
    seen_inputs: set[str] = set()
    for episode_id, raw_sample, abstract_sample, _ in sorted(candidates, key=lambda item: item[3], reverse=True):
        input_display = _primary_input_display(abstract_sample["task"]["input"])
        if input_display in seen_inputs:
            continue
        seen_inputs.add(input_display)
        selected.append(_build_chain_example(entry_function, raw_sample, abstract_sample))
        if len(selected) >= limit:
            break
    return selected


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
        "sympy.apart": "Decompose a rational expression into partial fractions in the chosen variable.",
        "sympy.trigsimp": "Apply direct trigonometric simplification to the current expression.",
        "sympy.powsimp": "Normalize multiplicative powers and combine compatible exponents.",
        "sympy.solveset": "Solve the equation as a set-valued problem over the default domain.",
        "sympy.factor_list": "Factor the polynomial while keeping multiplicities explicit in a structured result.",
        "sympy.linsolve": "Solve the linear system directly and return the solution set.",
        "sympy.linear_eq_to_matrix": "Convert the linear system into matrix form before downstream solving.",
        "sympy.groebner": "Construct a Groebner basis for the polynomial system under the given variable order.",
        "sympy.GroebnerBasis.reduce": "Reduce the target polynomial with respect to the computed Groebner basis.",
        "sympy.diff": "Differentiate the current expression with respect to the default variable.",
        "sympy.integrate": "Integrate the current expression with respect to the default variable.",
        "sympy.limit": "Evaluate the limiting value at the default expansion point.",
        "sympy.series": "Expand the current expression into a truncated series around the default point.",
        "sympy.residue": "Compute the residue at the requested pole.",
        "sympy.summation": "Evaluate the finite symbolic summation over the given bounds.",
        "sympy.collect": "Collect terms by the main symbol to expose grouped structure.",
        "sympy.expand_mul": "Expand multiplicative structure without changing the broader problem.",
        "sympy.expand_trig": "Expand trigonometric sums or multiples into simpler trig pieces.",
        "sympy.expand_log": "Expand logarithms into additive pieces when allowed.",
        "sympy.expand_power_base": "Distribute powers over multiplicative bases when valid.",
        "sympy.expand_power_exp": "Split additive exponents into separate power factors when valid.",
        "sympy.logcombine": "Merge compatible logarithmic terms into a compact form.",
        "sympy.hyperexpand": "Rewrite the hypergeometric form into a more explicit expression.",
        "sympy.combsimp": "Simplify factorial or binomial structure combinatorially.",
        "sympy.gammasimp": "Simplify gamma-function ratios or products.",
        "sympy.gcd": "Compute the greatest common divisor of the polynomial inputs.",
        "sympy.gcd_list": "Compute the common divisor shared by the full list of expressions.",
        "sympy.lcm": "Compute the least common multiple of the polynomial inputs.",
        "sympy.resultant": "Eliminate the main variable by taking the resultant.",
        "sympy.terms_gcd": "Extract a common factor shared across additive terms.",
        "sympy.reduce_inequalities": "Solve the inequality and return the admissible region.",
        "sympy.checksol": "Verify whether the proposed candidate satisfies the equation.",
        "sympy.diophantine": "Enumerate integer-valued solution families for the equation.",
        "sympy.nonlinsolve": "Solve the nonlinear system exactly as a structured solution set.",
        "sympy.nsolve": "Numerically refine a root from the supplied initial guess.",
        "sympy.solve_linear": "Solve the expression as a linear equation in the target variable.",
        "sympy.solve_undetermined_coeffs": "Match coefficients to solve for unknown symbolic parameters.",
        "sympy.solve_linear_system": "Solve the augmented linear system directly from matrix form.",
        "sympy.solve_linear_system_LU": "Solve the augmented linear system using LU decomposition.",
        "sympy.solve_poly_system": "Solve the polynomial system across all requested variables.",
        "sympy.unrad": "Rewrite the equation into a form with radicals removed.",
        "sympy.Interval": "Build the interval object directly from its endpoints.",
        "sympy.Union": "Combine the provided sets into a union.",
        "sympy.Complement": "Subtract the second set from the first one.",
        "sympy.simplify_logic": "Reduce the boolean expression to a simpler logical form.",
    }
    return reasons.get(tool_name, "Apply the next SymPy transformation indicated by the reference trajectory.")


def _sanitize_tool_steps(abstract_sample: dict | None) -> list[dict]:
    if not abstract_sample:
        return []
    task = abstract_sample["task"]["input"]
    symbol_names = _parse_symbol_names(task.get("symbol_names", {}).get("display"))
    steps = []
    pending_steps = []
    for index, (assistant_step, tool_step) in enumerate(
        zip(abstract_sample["gold_trajectory"][::2], abstract_sample["gold_trajectory"][1::2]),
        start=1,
    ):
        tool_name = assistant_step["assistant"]["tool_call"]["tool"]
        tool_args = assistant_step["assistant"]["tool_call"]["args"]
        current_handle = "$0" if not pending_steps else f"${len(pending_steps)}"
        clean_args = _canonical_step_args(tool_name, task, current_handle, symbol_names=symbol_names, tool_args=tool_args)
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
        prior_handle = "$0" if not deduped else f"${len(deduped)}"
        normalized_args = _rebind_primary_arg(step["tool"], dict(step["args"]), prior_handle)
        normalized["args"] = normalized_args
        deduped.append(normalized)
        previous = signature
    return deduped


def _canonical_step_args(
    tool_name: str,
    task_input: dict,
    current_handle: str,
    *,
    symbol_names: list[str] | None,
    tool_args: dict,
) -> dict:
    if tool_name in {
        "sympy.simplify",
        "sympy.trigsimp",
        "sympy.powsimp",
        "sympy.radsimp",
        "sympy.nsimplify",
        "sympy.collect",
        "sympy.cancel",
        "sympy.apart",
        "sympy.together",
        "sympy.factor",
        "sympy.expand",
        "sympy.expand_mul",
        "sympy.expand_log",
        "sympy.expand_trig",
        "sympy.expand_power_base",
        "sympy.expand_power_exp",
        "sympy.logcombine",
        "sympy.hyperexpand",
        "sympy.combsimp",
        "sympy.gammasimp",
        "sympy.factor_terms",
        "sympy.rational_simplify",
        "sympy.trig_simplify",
        "sympy.diff",
        "sympy.integrate",
        "sympy.limit",
        "sympy.series",
        "sympy.residue",
        "sympy.summation",
        "sympy.Poly",
        "sympy.factor_list",
        "sympy.sqf_part",
        "sympy.sqf_list",
        "sympy.discriminant",
        "sympy.roots",
        "sympy.solve",
        "sympy.solveset",
        "sympy.solve_linear",
        "sympy.unrad",
        "sympy.terms_gcd",
        "sympy.simplify_logic",
    }:
        args = {"expr": current_handle}
        if tool_name in {"sympy.solve", "sympy.solveset"} and symbol_names:
            args["symbol_names"] = symbol_names
        return args
    if tool_name in {"sympy.linsolve", "sympy.linear_eq_to_matrix", "sympy.groebner", "sympy.nonlinsolve", "sympy.solve_poly_system"}:
        args = {"system": current_handle}
        if symbol_names:
            args["symbol_names"] = symbol_names
        return args
    if tool_name == "sympy.GroebnerBasis.reduce":
        args = {"basis_exprs": current_handle}
        if "target_expr" in task_input:
            args["target_expr"] = task_input["target_expr"]["display"]
        if symbol_names:
            args["symbol_names"] = symbol_names
        return args
    if tool_name in {"sympy.gcd", "sympy.lcm", "sympy.resultant", "sympy.div", "sympy.rem"}:
        other = task_input.get("other", {}).get("display")
        return {"expr": current_handle, "other": other}
    if tool_name == "sympy.gcd_list":
        return {"exprs": current_handle}
    if tool_name in {"sympy.solve_linear_system", "sympy.solve_linear_system_LU"}:
        args = {"matrix_rows": current_handle}
        if symbol_names:
            args["symbol_names"] = symbol_names
        return args
    if tool_name == "sympy.solve_undetermined_coeffs":
        return {
            "expr": current_handle,
            "coeff_names": task_input.get("coeff_names", {}).get("display"),
        }
    if tool_name == "sympy.nsolve":
        return {
            "expr": current_handle,
            "symbol_name": task_input.get("symbol_name", {}).get("display"),
            "guess": task_input.get("guess", {}).get("display"),
        }
    if tool_name == "sympy.reduce_inequalities":
        return {"inequalities": current_handle}
    if tool_name == "sympy.checksol":
        return {
            "expr": current_handle,
            "symbol_name": task_input.get("symbol_name", {}).get("display"),
            "candidate": task_input.get("candidate", {}).get("display"),
        }
    if tool_name == "sympy.diophantine":
        return {"expr": current_handle}
    if tool_name == "sympy.residue":
        return {
            "expr": current_handle,
            "symbol_name": task_input.get("symbol_name", {}).get("display"),
            "point": task_input.get("point", {}).get("display"),
        }
    if tool_name == "sympy.summation":
        return {
            "expr": current_handle,
            "symbol_name": task_input.get("symbol_name", {}).get("display"),
            "lower": task_input.get("lower", {}).get("display"),
            "upper": task_input.get("upper", {}).get("display"),
        }
    if tool_name == "sympy.Interval":
        return {
            "start": task_input.get("start", {}).get("display"),
            "end": task_input.get("end", {}).get("display"),
        }
    if tool_name == "sympy.FiniteSet":
        return {"elements": current_handle}
    if tool_name in {"sympy.Union", "sympy.Intersection"}:
        return {"sets": current_handle}
    if tool_name == "sympy.Complement":
        return {
            "left": current_handle,
            "right": task_input.get("right", {}).get("display"),
        }
    if tool_name == "sympy.inspect_branches":
        args = {"expr": current_handle}
        if symbol_names:
            args["symbol_names"] = symbol_names
        return args
    if tool_name == "sympy.solve_branch":
        args = {"expr": current_handle}
        if symbol_names:
            args["symbol_names"] = symbol_names
        if "branch_id" in tool_args:
            args["branch_id"] = tool_args["branch_id"]
        return args
    return {"expr": current_handle}


def _rebind_primary_arg(tool_name: str, args: dict, prior_handle: str) -> dict:
    primary_by_tool = {
        "sympy.linsolve": "system",
        "sympy.linear_eq_to_matrix": "system",
        "sympy.groebner": "system",
        "sympy.nonlinsolve": "system",
        "sympy.solve_poly_system": "system",
        "sympy.GroebnerBasis.reduce": "basis_exprs",
        "sympy.gcd_list": "exprs",
        "sympy.solve_linear_system": "matrix_rows",
        "sympy.solve_linear_system_LU": "matrix_rows",
        "sympy.reduce_inequalities": "inequalities",
        "sympy.Interval": None,
        "sympy.FiniteSet": "elements",
        "sympy.Union": "sets",
        "sympy.Intersection": "sets",
        "sympy.Complement": "left",
    }
    primary_key = primary_by_tool.get(tool_name, "expr")
    if primary_key is None:
        return args
    args[primary_key] = prior_handle
    return args


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
        lines.append(f"Task: {example['task_goal']}")
        lines.append("")
        for idx, step in enumerate(example["ideal_tool_steps"], start=1):
            lines.append(f"Step {idx}")
            lines.append(f"- Reasoning: {step['reason']}")
            lines.append(f"- Tool call: `{step['tool']}` with args `{step['args']}`")
            lines.append(f"- Observation: `{step['result']}` = `{step['value']}`")
            lines.append("")
        lines.append(f"Expected answer: `{example['final_output']}`")
        lines.append("")
    return "\n".join(lines)
