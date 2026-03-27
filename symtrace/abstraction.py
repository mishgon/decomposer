from __future__ import annotations

import re

from .schemas import AbstractStep


ABSTRACT_TOOL_NAMES = {
    "sympy.simplify.simplify.simplify": "sympy.simplify",
    "sympy.simplify.trigsimp.trigsimp": "sympy.trigsimp",
    "sympy.simplify.powsimp.powsimp": "sympy.powsimp",
    "sympy.simplify.radsimp.radsimp": "sympy.radsimp",
    "sympy.simplify.simplify.nsimplify": "sympy.nsimplify",
    "sympy.simplify.combsimp.combsimp": "sympy.combsimp",
    "sympy.simplify.gammasimp.gammasimp": "sympy.gammasimp",
    "sympy.simplify.radsimp.collect": "sympy.collect",
    "sympy.core.function.expand_mul": "sympy.expand_mul",
    "sympy.simplify.trigsimp.expand_trig": "sympy.expand_trig",
    "sympy.core.function.expand_log": "sympy.expand_log",
    "sympy.core.function.expand_power_base": "sympy.expand_power_base",
    "sympy.core.function.expand_power_exp": "sympy.expand_power_exp",
    "sympy.simplify.simplify.logcombine": "sympy.logcombine",
    "sympy.polys.polytools.cancel": "sympy.cancel",
    "sympy.polys.partfrac.apart": "sympy.apart",
    "sympy.polys.rationaltools.together": "sympy.together",
    "sympy.core.exprtools.factor_terms": "sympy.factor_terms",
    "sympy.polys.polytools.factor": "sympy.factor",
    "sympy.polys.polytools.factor_list": "sympy.factor_list",
    "sympy.polys.polytools.Poly": "sympy.Poly",
    "sympy.polys.polytools.sqf_part": "sympy.sqf_part",
    "sympy.polys.polytools.sqf_list": "sympy.sqf_list",
    "sympy.polys.polytools.discriminant": "sympy.discriminant",
    "sympy.polys.polytools.gcd": "sympy.gcd",
    "sympy.polys.polytools.gcd_list": "sympy.gcd_list",
    "sympy.polys.polytools.lcm": "sympy.lcm",
    "sympy.polys.polytools.resultant": "sympy.resultant",
    "sympy.polys.polytools.terms_gcd": "sympy.terms_gcd",
    "sympy.polys.polytools.div": "sympy.div",
    "sympy.polys.polytools.rem": "sympy.rem",
    "sympy.polys.polytools._generic_factor": "sympy.factor_core",
    "sympy.polys.polytools._symbolic_factor": "sympy.factor_symbolic",
    "sympy.polys.polyroots.roots": "sympy.roots",
    "sympy.polys.polytools.groebner": "sympy.groebner",
    "sympy.polys.polytools.GroebnerBasis.reduce": "sympy.GroebnerBasis.reduce",
    "sympy.simplify.hyperexpand.hyperexpand": "sympy.hyperexpand",
    "sympy.core.function.expand": "sympy.expand",
    "sympy.core.function.diff": "sympy.diff",
    "sympy.integrals.integrals.integrate": "sympy.integrate",
    "sympy.series.limits.limit": "sympy.limit",
    "sympy.series.series.series": "sympy.series",
    "sympy.solvers.solvers.solve": "sympy.solve",
    "sympy.solvers.solveset.solveset": "sympy.solveset",
    "sympy.solvers.solveset.linsolve": "sympy.linsolve",
    "sympy.solvers.solveset.nonlinsolve": "sympy.nonlinsolve",
    "sympy.solvers.solvers.nsolve": "sympy.nsolve",
    "sympy.solvers.solveset.linear_eq_to_matrix": "sympy.linear_eq_to_matrix",
    "sympy.solvers.inequalities.reduce_inequalities": "sympy.reduce_inequalities",
    "sympy.solvers.solvers.checksol": "sympy.checksol",
    "sympy.solvers.diophantine.diophantine.diophantine": "sympy.diophantine",
    "sympy.solvers.solvers.solve_linear": "sympy.solve_linear",
    "sympy.solvers.solvers.solve_undetermined_coeffs": "sympy.solve_undetermined_coeffs",
    "sympy.solvers.solvers.solve_linear_system": "sympy.solve_linear_system",
    "sympy.solvers.solvers.solve_linear_system_LU": "sympy.solve_linear_system_LU",
    "sympy.solvers.polysys.solve_poly_system": "sympy.solve_poly_system",
    "sympy.solvers.solvers._solve": "sympy.solve_subproblem",
    "sympy.solvers.solvers._solve_system": "sympy.solve_system",
    "sympy.solvers.solvers.unrad": "sympy.unrad",
    "sympy.sets.sets.Interval": "sympy.Interval",
    "sympy.sets.sets.FiniteSet": "sympy.FiniteSet",
    "sympy.sets.sets.Union": "sympy.Union",
    "sympy.sets.sets.Intersection": "sympy.Intersection",
    "sympy.sets.sets.Complement": "sympy.Complement",
    "sympy.series.residues.residue": "sympy.residue",
    "sympy.concrete.summations.summation": "sympy.summation",
    "sympy.logic.boolalg.simplify_logic": "sympy.simplify_logic",
}

SKIP_IF_UNCHANGED = {
    "sympy.cancel",
    "sympy.powsimp",
    "sympy.trigsimp",
    "sympy.together",
    "sympy.factor_terms",
    "sympy.factor_list",
}

RATIONAL_CLUSTER = {
    "sympy.cancel",
    "sympy.together",
    "sympy.factor_terms",
    "sympy.powsimp",
    "sympy.hyperexpand",
}
TRIG_CLUSTER = {"sympy.trigsimp"}
SOLVE_CLUSTER = {"sympy.solve_subproblem", "sympy.solve_system", "sympy.solve_linear"}


def abstract_episode(raw_episode: dict) -> list[dict]:
    steps: list[dict] = []
    root_call_id = raw_episode["episode"].get("root_call_id")
    entry_tool = _entry_tool_name(raw_episode["episode"].get("entry_func_id"))
    raw_steps: list[dict] = []
    previous_signature: tuple[str, tuple[tuple[str, str], ...], str] | None = None
    for call_id in raw_episode["episode"]["call_ids"]:
        call = raw_episode["calls"][call_id]
        tool_name = ABSTRACT_TOOL_NAMES.get(call["func_id"])
        if not tool_name:
            continue
        if call_id == root_call_id and tool_name == "sympy.simplify":
            continue
        step_args = _extract_args(call["inputs"])
        output = call["output"] or {"display": "None"}
        if _should_skip_step(tool_name, step_args, output, preserve=call_id == root_call_id and tool_name == entry_tool):
            continue
        signature = (tool_name, tuple(sorted(step_args.items())), output.get("display", "None"))
        if signature == previous_signature:
            continue
        raw_steps.append({"tool": tool_name, "args": step_args, "value": output})
        previous_signature = signature
    macro_steps = _compress_steps(raw_steps, preserve_root_tool=entry_tool)
    for handle_index, step in enumerate(macro_steps, start=1):
        steps.append(
            AbstractStep(
                tool=step["tool"],
                args=step["args"],
                result_handle=f"${handle_index}",
                value=step["value"],
            ).to_dict()
        )
    if not steps:
        root_call = raw_episode["calls"].get(root_call_id) if root_call_id else None
        if root_call:
            root_tool = ABSTRACT_TOOL_NAMES.get(root_call["func_id"])
            root_args = _extract_args(root_call["inputs"])
            root_output = root_call["output"] or {"display": "None"}
            if root_tool and root_tool == "sympy.simplify":
                input_display = root_args.get("arg0") or root_args.get("expr")
                if input_display and input_display != root_output.get("display"):
                    steps.append(
                        AbstractStep(
                            tool=root_tool,
                            args=root_args,
                            result_handle="$1",
                            value=root_output,
                        ).to_dict()
                    )
    return steps


def _extract_args(inputs: dict) -> dict:
    args = {}
    for name, value in inputs.items():
        args[name] = value.get("display")
    return args


def _entry_tool_name(entry_func_id: str | None) -> str | None:
    if not entry_func_id:
        return None
    return f"sympy.{entry_func_id}"


def _should_skip_step(tool_name: str, args: dict, output: dict, *, preserve: bool = False) -> bool:
    if preserve:
        return False
    input_display = args.get("arg0") or args.get("expr")
    output_display = output.get("display")
    if tool_name in SKIP_IF_UNCHANGED and input_display and input_display == output_display:
        return True
    return False


def _compress_steps(raw_steps: list[dict], *, preserve_root_tool: str | None = None) -> list[dict]:
    compressed: list[dict] = []
    idx = 0
    while idx < len(raw_steps):
        step = raw_steps[idx]
        tool = step["tool"]

        if idx == 0 and tool == preserve_root_tool:
            compressed.append(step)
            idx += 1
            continue

        if tool in RATIONAL_CLUSTER:
            idx, compressed_step = _consume_cluster(
                raw_steps,
                idx,
                RATIONAL_CLUSTER,
                macro_tool="sympy.rational_simplify",
            )
            compressed.append(compressed_step)
            continue

        if tool in TRIG_CLUSTER:
            idx, compressed_step = _consume_cluster(
                raw_steps,
                idx,
                TRIG_CLUSTER,
                macro_tool="sympy.trig_simplify",
            )
            compressed.append(compressed_step)
            continue

        if tool in SOLVE_CLUSTER:
            idx, compressed_steps = _consume_solve_cluster(raw_steps, idx)
            compressed.extend(compressed_steps)
            continue

        if tool in {"sympy.factor_core", "sympy.factor_symbolic", "sympy.factor_list"}:
            idx, compressed_step = _consume_cluster(
                raw_steps,
                idx,
                {"sympy.factor_core", "sympy.factor_symbolic", "sympy.factor_list"},
                macro_tool="sympy.factor_decompose",
            )
            compressed.append(compressed_step)
            continue

        compressed.append(step)
        idx += 1

    return _dedupe_macro_steps(compressed)


def _consume_cluster(raw_steps: list[dict], start_idx: int, members: set[str], *, macro_tool: str) -> tuple[int, dict]:
    first = raw_steps[start_idx]
    end_idx = start_idx
    last = first
    while end_idx < len(raw_steps) and raw_steps[end_idx]["tool"] in members:
        last = raw_steps[end_idx]
        end_idx += 1
    return end_idx, {"tool": macro_tool, "args": first["args"], "value": last["value"]}


def _dedupe_macro_steps(steps: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    previous_signature: tuple[str, tuple[tuple[str, str], ...], str] | None = None
    for step in steps:
        signature = (
            step["tool"],
            tuple(sorted(step["args"].items())),
            step["value"].get("display", "None"),
        )
        if signature == previous_signature:
            continue
        deduped.append(step)
        previous_signature = signature
    return deduped


def _consume_solve_cluster(raw_steps: list[dict], start_idx: int) -> tuple[int, list[dict]]:
    cluster: list[dict] = []
    end_idx = start_idx
    while end_idx < len(raw_steps) and raw_steps[end_idx]["tool"] in SOLVE_CLUSTER:
        cluster.append(raw_steps[end_idx])
        end_idx += 1

    first = cluster[0]
    branch_candidates = _extract_branch_candidates(cluster)
    if not branch_candidates:
        return end_idx, [{"tool": "sympy.solve_branch", "args": first["args"], "value": cluster[-1]["value"]}]

    compressed: list[dict] = [
        {
            "tool": "sympy.inspect_branches",
            "args": first["args"],
            "value": {
                "display": str(
                    [
                        {"branch_id": f"b{idx}", "binding": branch, "resolved_value": branch}
                        for idx, branch in enumerate(branch_candidates)
                    ]
                )
            },
        }
    ]
    for idx, branch in enumerate(branch_candidates):
        compressed.append(
            {
                "tool": "sympy.solve_branch",
                "args": {**first["args"], "branch_id": f"b{idx}"},
                "value": {"display": str({"branch_id": f"b{idx}", "binding": branch, "resolved_value": branch})},
            }
        )
    return end_idx, compressed


def _extract_branch_candidates(cluster: list[dict]) -> list[str]:
    candidates: list[str] = []
    for step in cluster:
        if step["tool"] not in {"sympy.solve_subproblem", "sympy.solve_system"}:
            continue
        display = step["value"].get("display", "")
        if display.startswith("[]"):
            continue
        for branch in _extract_mapping_strings(display):
            if branch not in candidates:
                candidates.append(branch)
    return candidates


def _extract_mapping_strings(display: str) -> list[str]:
    return re.findall(r"\{[^{}]+\}", display)
