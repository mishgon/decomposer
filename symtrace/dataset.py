from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .episode_runner import EpisodeRunner
from .generators import make_apart_seed, make_equation_seed, make_expr_seed, make_solveset_seed
from .generators import (
    make_diff_seed,
    make_discriminant_seed,
    make_division_seed,
    make_groebner_seed,
    make_integrate_seed,
    make_limit_seed,
    make_linear_system_seed,
    make_poly_factor_seed,
    make_series_seed,
)
from .generators import (
    make_checksol_seed,
    make_collect_seed,
    make_complement_seed,
    make_diophantine_seed,
    make_expand_log_seed,
    make_expand_trig_seed,
    make_finiteset_seed,
    make_inequality_seed,
    make_intersection_seed,
    make_interval_seed,
    make_logcombine_seed,
    make_union_seed,
)
from .generators import (
    make_combsimp_seed,
    make_expand_mul_seed,
    make_expand_power_base_seed,
    make_expand_power_exp_seed,
    make_gammasimp_seed,
    make_hyperexpand_seed,
    make_nonlinsolve_seed,
    make_nsimplify_seed,
    make_nsolve_seed,
    make_radsimp_seed,
    make_residue_seed,
    make_simplify_logic_seed,
    make_solve_linear_seed,
    make_solve_linear_system_seed,
    make_solve_poly_system_seed,
    make_solve_undetermined_coeffs_seed,
    make_summation_seed,
    make_terms_gcd_seed,
    make_unrad_seed,
)
from .index_repo import function_def_from_callable
from .reporting import write_dataset_reports
from .sft.export_traces import export_training_example
from .sympy_wrappers.entrypoints import TOOL_SPECS


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


def build_demo_dataset(num_expr: int = 4, num_solve: int = 4) -> list[dict]:
    runner = EpisodeRunner()
    episodes: list[dict] = []
    for seed in range(num_expr):
        for entrypoint in ("simplify", "trigsimp", "powsimp", "cancel", "apart", "together", "factor", "expand"):
            expr = make_apart_seed(seed) if entrypoint == "apart" else make_expr_seed(seed)
            episodes.append(
                runner.run_episode(
                    episode_id=f"{entrypoint}_{seed}",
                    entrypoint_name=entrypoint,
                    kwargs={"expr": expr},
                    input_seed=seed,
                )
            )
        for entrypoint, expr in (
            ("nsimplify", make_nsimplify_seed(seed)),
            ("radsimp", make_radsimp_seed(seed)),
            ("hyperexpand", make_hyperexpand_seed(seed)),
            ("combsimp", make_combsimp_seed(seed)),
            ("gammasimp", make_gammasimp_seed(seed)),
            ("collect", make_collect_seed(seed)),
            ("expand_mul", make_expand_mul_seed(seed)),
            ("expand_trig", make_expand_trig_seed(seed)),
            ("expand_log", make_expand_log_seed(seed)),
            ("expand_power_base", make_expand_power_base_seed(seed)),
            ("expand_power_exp", make_expand_power_exp_seed(seed)),
            ("logcombine", make_logcombine_seed(seed)),
            ("terms_gcd", make_terms_gcd_seed(seed)),
            ("simplify_logic", make_simplify_logic_seed(seed)),
        ):
            episodes.append(
                runner.run_episode(
                    episode_id=f"{entrypoint}_{seed}",
                    entrypoint_name=entrypoint,
                    kwargs={"expr": expr},
                    input_seed=seed,
                )
            )
        poly_expr = make_poly_factor_seed(seed)
        for entrypoint in ("Poly", "factor_list", "sqf_part", "sqf_list", "roots"):
            episodes.append(
                runner.run_episode(
                    episode_id=f"{entrypoint}_{seed}",
                    entrypoint_name=entrypoint,
                    kwargs={"expr": poly_expr},
                    input_seed=seed,
                )
            )
        discr_expr = make_discriminant_seed(seed)
        episodes.append(
            runner.run_episode(
                episode_id=f"discriminant_{seed}",
                entrypoint_name="discriminant",
                kwargs={"expr": discr_expr},
                input_seed=seed,
            )
        )
        for entrypoint, kwargs in (
            ("gcd", {"expr": "x**2 - 1", "other": "x**2 - 3*x + 2"}),
            ("gcd_list", {"exprs": "[x**2 - 1, x**2 - 3*x + 2]"}),
            ("lcm", {"expr": "x - 1", "other": "x + 1"}),
            ("resultant", {"expr": "x**2 - 1", "other": "x - 1"}),
        ):
            episodes.append(
                runner.run_episode(
                    episode_id=f"{entrypoint}_{seed}",
                    entrypoint_name=entrypoint,
                    kwargs=kwargs,
                    input_seed=seed,
                )
            )
        div_expr, div_other = make_division_seed(seed)
        for entrypoint in ("div", "rem"):
            episodes.append(
                runner.run_episode(
                    episode_id=f"{entrypoint}_{seed}",
                    entrypoint_name=entrypoint,
                    kwargs={"expr": div_expr, "other": div_other},
                    input_seed=seed,
                )
            )
        groebner_system, groebner_symbols = make_groebner_seed(seed)
        episodes.append(
            runner.run_episode(
                episode_id=f"GroebnerBasis.reduce_{seed}",
                entrypoint_name="GroebnerBasis.reduce",
                kwargs={"basis_exprs": groebner_system, "target_expr": "x**2*y - 1", "symbol_names": groebner_symbols},
                input_seed=seed,
            )
        )
        for entrypoint, expr in (
            ("diff", make_diff_seed(seed)),
            ("integrate", make_integrate_seed(seed)),
            ("limit", make_limit_seed(seed)),
            ("series", make_series_seed(seed)),
        ):
            episodes.append(
                runner.run_episode(
                    episode_id=f"{entrypoint}_{seed}",
                    entrypoint_name=entrypoint,
                    kwargs={"expr": expr},
                    input_seed=seed,
                )
            )
        expr, symbol_name, point = make_residue_seed(seed)
        episodes.append(
            runner.run_episode(
                episode_id=f"residue_{seed}",
                entrypoint_name="residue",
                kwargs={"expr": expr, "symbol_name": symbol_name, "point": point},
                input_seed=seed,
            )
        )
        expr, symbol_name, lower, upper = make_summation_seed(seed)
        episodes.append(
            runner.run_episode(
                episode_id=f"summation_{seed}",
                entrypoint_name="summation",
                kwargs={"expr": expr, "symbol_name": symbol_name, "lower": lower, "upper": upper},
                input_seed=seed,
            )
        )
        start, end = make_interval_seed(seed)
        episodes.append(
            runner.run_episode(
                episode_id=f"Interval_{seed}",
                entrypoint_name="Interval",
                kwargs={"start": start, "end": end},
                input_seed=seed,
            )
        )
        episodes.append(
            runner.run_episode(
                episode_id=f"FiniteSet_{seed}",
                entrypoint_name="FiniteSet",
                kwargs={"elements": make_finiteset_seed(seed)},
                input_seed=seed,
            )
        )
        episodes.append(
            runner.run_episode(
                episode_id=f"Union_{seed}",
                entrypoint_name="Union",
                kwargs={"sets": make_union_seed(seed)},
                input_seed=seed,
            )
        )
        episodes.append(
            runner.run_episode(
                episode_id=f"Intersection_{seed}",
                entrypoint_name="Intersection",
                kwargs={"sets": make_intersection_seed(seed)},
                input_seed=seed,
            )
        )
        left, right = make_complement_seed(seed)
        episodes.append(
            runner.run_episode(
                episode_id=f"Complement_{seed}",
                entrypoint_name="Complement",
                kwargs={"left": left, "right": right},
                input_seed=seed,
            )
        )
    for seed in range(num_solve):
        for entrypoint in ("solve", "solveset"):
            expr, symbol_names = make_solveset_seed(seed) if entrypoint == "solveset" else make_equation_seed(seed)
            episodes.append(
                runner.run_episode(
                    episode_id=f"{entrypoint}_{seed}",
                    entrypoint_name=entrypoint,
                    kwargs={"expr": expr, "symbol_names": symbol_names},
                    input_seed=seed,
                )
            )
        system, symbol_names = make_linear_system_seed(seed)
        for entrypoint in ("linsolve", "linear_eq_to_matrix"):
            episodes.append(
                runner.run_episode(
                    episode_id=f"{entrypoint}_{seed}",
                    entrypoint_name=entrypoint,
                    kwargs={"system": system, "symbol_names": symbol_names},
                    input_seed=seed,
                )
            )
        system, symbol_names = make_groebner_seed(seed)
        episodes.append(
            runner.run_episode(
                episode_id=f"groebner_{seed}",
                entrypoint_name="groebner",
                kwargs={"system": system, "symbol_names": symbol_names},
                input_seed=seed,
            )
        )
        episodes.append(
            runner.run_episode(
                episode_id=f"reduce_inequalities_{seed}",
                entrypoint_name="reduce_inequalities",
                kwargs={"inequalities": make_inequality_seed(seed)},
                input_seed=seed,
            )
        )
        expr, symbol_name, candidate = make_checksol_seed(seed)
        episodes.append(
            runner.run_episode(
                episode_id=f"checksol_{seed}",
                entrypoint_name="checksol",
                kwargs={"expr": expr, "symbol_name": symbol_name, "candidate": candidate},
                input_seed=seed,
            )
        )
        episodes.append(
            runner.run_episode(
                episode_id=f"diophantine_{seed}",
                entrypoint_name="diophantine",
                kwargs={"expr": make_diophantine_seed(seed)},
                input_seed=seed,
            )
        )
        system, symbol_names = make_nonlinsolve_seed(seed)
        episodes.append(
            runner.run_episode(
                episode_id=f"nonlinsolve_{seed}",
                entrypoint_name="nonlinsolve",
                kwargs={"system": system, "symbol_names": symbol_names},
                input_seed=seed,
            )
        )
        expr, symbol_name, guess = make_nsolve_seed(seed)
        episodes.append(
            runner.run_episode(
                episode_id=f"nsolve_{seed}",
                entrypoint_name="nsolve",
                kwargs={"expr": expr, "symbol_name": symbol_name, "guess": guess},
                input_seed=seed,
            )
        )
        episodes.append(
            runner.run_episode(
                episode_id=f"solve_linear_{seed}",
                entrypoint_name="solve_linear",
                kwargs={"expr": make_solve_linear_seed(seed)},
                input_seed=seed,
            )
        )
        matrix_rows, symbol_names = make_solve_linear_system_seed(seed)
        for entrypoint in ("solve_linear_system", "solve_linear_system_LU"):
            episodes.append(
                runner.run_episode(
                    episode_id=f"{entrypoint}_{seed}",
                    entrypoint_name=entrypoint,
                    kwargs={"matrix_rows": matrix_rows, "symbol_names": symbol_names},
                    input_seed=seed,
                )
            )
        system, symbol_names = make_solve_poly_system_seed(seed)
        episodes.append(
            runner.run_episode(
                episode_id=f"solve_poly_system_{seed}",
                entrypoint_name="solve_poly_system",
                kwargs={"system": system, "symbol_names": symbol_names},
                input_seed=seed,
            )
        )
        expr, coeff_names = make_solve_undetermined_coeffs_seed(seed)
        episodes.append(
            runner.run_episode(
                episode_id=f"solve_undetermined_coeffs_{seed}",
                entrypoint_name="solve_undetermined_coeffs",
                kwargs={"expr": expr, "coeff_names": str(coeff_names)},
                input_seed=seed,
            )
        )
        episodes.append(
            runner.run_episode(
                episode_id=f"unrad_{seed}",
                entrypoint_name="unrad",
                kwargs={"expr": make_unrad_seed(seed)},
                input_seed=seed,
            )
        )
    return episodes


def export_demo_corpus(output_dir: str | Path, num_expr: int = 4, num_solve: int = 4) -> dict[str, int]:
    output = Path(output_dir)
    raw_dir = output / "raw_traces"
    abstract_path = output / "abstract_traces" / "train.jsonl"
    function_catalog_path = output / "function_catalog.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)
    abstract_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = [
        function_def_from_callable(
            spec.target,
            tool_id=spec.tool_id,
            kind=spec.kind,
            owner=spec.owner,
            family=spec.family,
            priority=spec.priority,
            callability=spec.callability,
            semantic_score=spec.semantic_score,
            returns_structured_object=spec.returns_structured_object,
            likely_trace_depth=spec.likely_trace_depth,
        )
        for spec in TOOL_SPECS
    ]
    with function_catalog_path.open("w", encoding="utf-8") as handle:
        for item in catalog:
            handle.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")

    episodes = build_demo_dataset(num_expr=num_expr, num_solve=num_solve)
    kept = 0
    for episode in episodes:
        episode_id = episode["episode"]["episode_id"]
        family_dir = raw_dir / episode["episode"]["task_family"]
        family_dir.mkdir(parents=True, exist_ok=True)
        raw_path = family_dir / f"{episode_id}.json"
        raw_path.write_text(json.dumps(episode, sort_keys=True, indent=2), encoding="utf-8")
        if episode["episode"]["status"] == "ok":
            export_training_example(episode, abstract_path)
            kept += 1
    summary = summarize_corpus(output)
    summary.update({"episodes": len(episodes), "training_examples": kept, "catalog_entries": len(catalog)})
    (output / "debug_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    (output / "debug_samples.json").write_text(
        json.dumps(sample_corpus(output), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    write_dataset_reports(output)
    return summary


def summarize_corpus(output_dir: str | Path) -> dict:
    root = Path(output_dir)
    raw_files = sorted((root / "raw_traces").rglob("*.json"))
    abstract_path = root / "abstract_traces" / "train.jsonl"
    status = Counter()
    patterns = Counter()
    family_calls = defaultdict(list)
    trajectory_lengths = Counter()

    for path in raw_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        episode = record["episode"]
        status[episode["status"]] += 1
        patterns[episode["pattern_label"]] += 1
        family_calls[episode["task_family"]].append(episode["num_calls"])

    if abstract_path.exists():
        for line in abstract_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = f"{record['task']['entry_function']}:{len(record['gold_trajectory']) // 2}"
            trajectory_lengths[key] += 1

    family_summary = {
        family: {
            "n": len(counts),
            "min": min(counts),
            "max": max(counts),
            "avg": round(sum(counts) / len(counts), 2),
        }
        for family, counts in sorted(family_calls.items())
    }
    return {
        "status_counts": dict(status),
        "pattern_counts": dict(patterns),
        "family_num_calls": family_summary,
        "trajectory_length_counts": dict(sorted(trajectory_lengths.items())),
    }


def sample_corpus(output_dir: str | Path, *, num_raw: int = 5, num_abstract: int = 5) -> dict:
    root = Path(output_dir)
    raw_files = sorted((root / "raw_traces").rglob("*.json"))
    abstract_path = root / "abstract_traces" / "train.jsonl"

    raw_samples = []
    for path in sorted(
        raw_files,
        key=lambda p: json.loads(p.read_text(encoding="utf-8"))["episode"]["num_calls"],
        reverse=True,
    )[:num_raw]:
        record = json.loads(path.read_text(encoding="utf-8"))
        episode = record["episode"]
        raw_samples.append(
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

    abstract_records = []
    if abstract_path.exists():
        for line in abstract_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["gold_trajectory"]:
                abstract_records.append(record)
    abstract_samples = sorted(
        abstract_records,
        key=lambda record: (len(record["gold_trajectory"]), record["task"]["entry_function"]),
        reverse=True,
    )[:num_abstract]

    return {"raw_samples": raw_samples, "abstract_samples": abstract_samples}
