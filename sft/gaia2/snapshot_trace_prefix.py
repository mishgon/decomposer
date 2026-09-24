"""Materialize an immutable, compact GAIA2 trace-prefix snapshot for SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gyms.gaia2.partition import (
    SPLIT_MANIFEST_RELPATH,
    SPLIT_MANIFEST_SHA256,
    partition_scenario_ids,
)
from gyms.gaia2.run import validate_trace_round


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def create_trace_prefix_snapshot(
    source: Path,
    output: Path,
    *,
    logical_rollout_numbers: Sequence[int],
    scenario_ids: Sequence[str],
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    logical_rollout_numbers = tuple(logical_rollout_numbers)
    scenario_ids = tuple(scenario_ids)
    if output.exists():
        raise FileExistsError(f"Immutable trace snapshot already exists: {output}")
    if (
        not logical_rollout_numbers
        or tuple(sorted(set(logical_rollout_numbers))) != logical_rollout_numbers
    ):
        raise ValueError("Logical rollout numbers must be unique, positive, and sorted")
    if any(number <= 0 for number in logical_rollout_numbers):
        raise ValueError("Logical rollout numbers must be positive")
    if not scenario_ids or len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Scenario IDs must be non-empty and unique")

    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Incomplete snapshot temporary exists: {temporary}")
    temporary.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, Any]] = {}
    round_metrics: dict[str, Any] = {}
    try:
        for logical in logical_rollout_numbers:
            round_name = f"round_{logical:02d}"
            round_directory = source / round_name
            marker_path = round_directory / ".round_done.json"
            output_path = round_directory / "output.jsonl"
            if not marker_path.is_file():
                raise FileNotFoundError(
                    f"Missing completed-round marker: {marker_path}"
                )
            metrics, round_records = validate_trace_round(
                round_directory,
                trace_directory=source,
                logical_rollout_number=logical,
                scenario_ids=scenario_ids,
            )
            round_metrics[round_name] = metrics
            source_files[f"{round_name}/.round_done.json"] = _file_identity(marker_path)
            source_files[f"{round_name}/output.jsonl"] = _file_identity(output_path)
            for record in round_records:
                compact = {
                    key: value
                    for key, value in record.items()
                    if key not in {"hf_trace", "lite_trace", "output_jsonl"}
                }
                sidecar = record.get("sidecar")
                if float(record["reward"]) == 1.0:
                    if sidecar is None:
                        raise FileNotFoundError(
                            "Passing trace manifest row has no sidecar path: "
                            f"{record['scenario_id']} r{logical:02d}"
                        )
                    source_sidecar = source / sidecar
                    target_sidecar = temporary / sidecar
                    if not source_sidecar.is_file():
                        raise FileNotFoundError(
                            f"Passing trace is missing its sidecar: {source_sidecar}"
                        )
                    target_sidecar.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_sidecar, target_sidecar)
                else:
                    compact["sidecar"] = None
                records.append(compact)

        identities = {
            (record["scenario_id"], record["logical_rollout_number"])
            for record in records
        }
        expected_attempts = len(scenario_ids) * len(logical_rollout_numbers)
        per_scenario = Counter(record["scenario_id"] for record in records)
        if len(records) != expected_attempts or len(identities) != expected_attempts:
            raise ValueError("Trace prefix is not a complete unique scenario/run grid")
        if set(per_scenario.values()) != {len(logical_rollout_numbers)}:
            raise ValueError("Trace prefix does not have equal rollout coverage")
        records.sort(
            key=lambda row: (row["logical_rollout_number"], row["scenario_id"])
        )
        manifest_path = temporary / "trace_manifest.jsonl"
        _write_jsonl(manifest_path, records)
        rewards = [float(record["reward"]) for record in records]
        marker: dict[str, Any] = {
            "schema_version": 1,
            "state": "complete",
            "kind": "decomposer",
            "purpose": "sft-trace-prefix-snapshot",
            "decomposer_system_prompt_profile": "teacher",
            "scenario_count": len(scenario_ids),
            "num_repeats": len(logical_rollout_numbers),
            "logical_rollout_numbers": list(logical_rollout_numbers),
            "attempted_rollouts": len(records),
            "unique_attempted_rollouts": len(identities),
            "passed_rollouts": sum(reward == 1.0 for reward in rewards),
            "failed_rollouts": sum(reward == 0.0 for reward in rewards),
            "source_trace_directory": str(source),
            "source_files": dict(sorted(source_files.items())),
            "split_manifest": {
                "path": SPLIT_MANIFEST_RELPATH,
                "sha256": SPLIT_MANIFEST_SHA256,
            },
            "rounds": round_metrics,
            "trace_manifest": _file_identity(manifest_path),
            "finished_at": datetime.now(UTC).isoformat(),
        }
        _write_json(temporary / ".trace_done.json", marker)
        os.rename(temporary, output)
        return marker
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--logical-rollout-number",
        type=_positive_int,
        action="append",
        required=True,
    )
    parser.add_argument("--partition", choices=("train",), default="train")
    args = parser.parse_args(argv)
    marker = create_trace_prefix_snapshot(
        args.source,
        args.output,
        logical_rollout_numbers=args.logical_rollout_number,
        scenario_ids=partition_scenario_ids(args.partition),
    )
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
