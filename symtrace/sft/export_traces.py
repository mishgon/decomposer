from __future__ import annotations

import json
from pathlib import Path

from ..abstraction import abstract_episode
from ..schemas import TrainingExample


def _display(input_payload: dict, key: str) -> str | None:
    value = input_payload.get(key)
    if isinstance(value, dict):
        return value.get("display")
    return None


def _task_goal(entry_function: str, input_payload: dict) -> str:
    expr = _display(input_payload, "expr")
    system = _display(input_payload, "system")
    symbol_names = _display(input_payload, "symbol_names")
    mapping = {
        "simplify": f"Simplify `{expr}`.",
        "trigsimp": f"Apply trigonometric simplification to `{expr}`.",
        "powsimp": f"Normalize powers in `{expr}`.",
        "radsimp": f"Simplify radicals in `{expr}`.",
        "nsimplify": f"Convert `{expr}` to a simpler exact symbolic form.",
        "hyperexpand": f"Expand the hypergeometric expression `{expr}`.",
        "combsimp": f"Simplify the combinatorial expression `{expr}`.",
        "gammasimp": f"Simplify gamma functions in `{expr}`.",
        "cancel": f"Cancel common factors in `{expr}`.",
        "apart": f"Decompose `{expr}` into partial fractions.",
        "together": f"Combine rational terms in `{expr}` into a single fraction.",
        "factor": f"Factor `{expr}`.",
        "expand": f"Expand `{expr}`.",
        "collect": f"Collect `{expr}` by the main symbol.",
        "expand_mul": f"Expand products in `{expr}`.",
        "expand_trig": f"Expand the trigonometric expression `{expr}`.",
        "expand_log": f"Expand logarithms in `{expr}`.",
        "expand_power_base": f"Expand power bases in `{expr}`.",
        "expand_power_exp": f"Expand exponents in `{expr}`.",
        "logcombine": f"Combine logarithms in `{expr}`.",
        "Poly": f"Construct a polynomial object from `{expr}`.",
        "factor_list": f"Factor `{expr}` and keep multiplicities explicit.",
        "sqf_part": f"Extract the square-free part of `{expr}`.",
        "sqf_list": f"Compute the square-free factorization of `{expr}`.",
        "discriminant": f"Compute the discriminant of `{expr}`.",
        "gcd": f"Compute the gcd of `{expr}` and `{_display(input_payload, 'other')}`.",
        "gcd_list": f"Compute the gcd across `{_display(input_payload, 'exprs')}`.",
        "lcm": f"Compute the lcm of `{expr}` and `{_display(input_payload, 'other')}`.",
        "resultant": f"Eliminate the main variable between `{expr}` and `{_display(input_payload, 'other')}` via the resultant.",
        "terms_gcd": f"Extract the gcd of terms in `{expr}`.",
        "roots": f"Find the roots of `{expr}`.",
        "div": f"Divide `{expr}` by `{_display(input_payload, 'other')}` and return quotient and remainder.",
        "rem": f"Compute the remainder of `{expr}` divided by `{_display(input_payload, 'other')}`.",
        "groebner": f"Compute a Groebner basis for `{system}` over `{symbol_names}`.",
        "GroebnerBasis.reduce": f"Reduce `{_display(input_payload, 'target_expr')}` modulo the Groebner basis of `{_display(input_payload, 'basis_exprs')}`.",
        "diff": f"Differentiate `{expr}` with respect to the main variable.",
        "integrate": f"Integrate `{expr}` with respect to the main variable.",
        "limit": f"Compute the limit of `{expr}` at the default point.",
        "series": f"Compute a truncated series expansion of `{expr}` around the default point.",
        "residue": f"Compute the residue of `{expr}` at `{_display(input_payload, 'point')}`.",
        "summation": f"Evaluate the summation of `{expr}` from `{_display(input_payload, 'lower')}` to `{_display(input_payload, 'upper')}`.",
        "solve": f"Solve `{expr}` for `{symbol_names}`.",
        "solveset": f"Solve `{expr}` as a set-valued equation for `{symbol_names}`.",
        "linsolve": f"Solve the linear system `{system}` for `{symbol_names}`.",
        "nonlinsolve": f"Solve the nonlinear system `{system}` for `{symbol_names}`.",
        "nsolve": f"Numerically solve `{expr}` starting from `{_display(input_payload, 'guess')}`.",
        "solve_linear": f"Solve the linear expression `{expr}` for the main variable.",
        "solve_undetermined_coeffs": f"Solve for the undetermined coefficients in `{expr}`.",
        "solve_linear_system": f"Solve the augmented matrix `{_display(input_payload, 'matrix_rows')}` for `{symbol_names}`.",
        "solve_linear_system_LU": f"Solve the augmented matrix `{_display(input_payload, 'matrix_rows')}` using LU decomposition.",
        "solve_poly_system": f"Solve the polynomial system `{system}` for `{symbol_names}`.",
        "linear_eq_to_matrix": f"Convert the linear system `{system}` over `{symbol_names}` into matrix form.",
        "reduce_inequalities": f"Reduce the inequality `{_display(input_payload, 'inequalities')}`.",
        "checksol": f"Check whether `{_display(input_payload, 'candidate')}` solves `{expr}`.",
        "diophantine": f"Find integer solutions to `{expr}`.",
        "unrad": f"Remove radicals from `{expr}`.",
        "Interval": f"Construct the interval from `{_display(input_payload, 'start')}` to `{_display(input_payload, 'end')}`.",
        "FiniteSet": f"Construct a finite set from `{_display(input_payload, 'elements')}`.",
        "Union": f"Take the union of `{_display(input_payload, 'sets')}`.",
        "Intersection": f"Take the intersection of `{_display(input_payload, 'sets')}`.",
        "Complement": f"Compute the complement of `{_display(input_payload, 'left')}` minus `{_display(input_payload, 'right')}`.",
        "simplify_logic": f"Simplify the boolean expression `{expr}`.",
    }
    return mapping.get(entry_function, "Apply the requested SymPy operation and return the exact symbolic result.")


def export_training_example(raw_episode: dict, output_path: str | Path) -> dict:
    episode = raw_episode["episode"]
    abstract_steps = abstract_episode(raw_episode)
    example = TrainingExample(
        task={
            "entry_function": episode["entry_func_id"],
            "input": episode["input"],
            "goal": _task_goal(episode["entry_func_id"], episode["input"]),
        },
        tools=sorted({step["tool"] for step in abstract_steps}),
        gold_trajectory=_to_dialogue(abstract_steps),
        final_answer=episode["final_output"],
        metadata={
            "episode_id": episode["episode_id"],
            "pattern_label": episode["pattern_label"],
            "status": episode["status"],
        },
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")
    return example.to_dict()


def _to_dialogue(steps: list[dict]) -> list[dict]:
    trajectory: list[dict] = []
    for step in steps:
        trajectory.append({"assistant": {"tool_call": {"tool": step["tool"], "args": step["args"]}}})
        trajectory.append({"tool": {"result": step["result_handle"], "value": step["value"]["display"]}})
    return trajectory
