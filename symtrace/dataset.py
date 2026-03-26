from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from sympy import expand, factor, simplify, solve

from .episode_runner import EpisodeRunner
from .generators import make_equation_seed, make_expr_seed
from .index_repo import build_function_catalog
from .reporting import write_dataset_reports
from .sft.export_traces import export_training_example


def build_demo_dataset(num_expr: int = 4, num_solve: int = 4) -> list[dict]:
    runner = EpisodeRunner()
    episodes: list[dict] = []
    for seed in range(num_expr):
        expr = make_expr_seed(seed)
        for entrypoint in ("simplify", "expand", "factor"):
            episodes.append(
                runner.run_episode(
                    episode_id=f"{entrypoint}_{seed}",
                    entrypoint_name=entrypoint,
                    kwargs={"expr": expr},
                    input_seed=seed,
                )
            )
    for seed in range(num_solve):
        expr, symbol_names = make_equation_seed(seed)
        episodes.append(
            runner.run_episode(
                episode_id=f"solve_{seed}",
                entrypoint_name="solve",
                kwargs={"expr": expr, "symbol_names": symbol_names},
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

    catalog = build_function_catalog([simplify, factor, expand, solve])
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
