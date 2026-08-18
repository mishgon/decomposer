"""Prepare Workplace Assistant evaluation inputs or canonical SFT datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gyms.workplace_assistant.experiments import (  # noqa: E402
    DATA_DIR,
    PROJECT_VENV,
    SFT_OUTPUT_ROOT,
    SPLITS,
    SPLIT_ROWS,
    UV_BIN,
    UV_CACHE,
    Experiment,
    SimpleExperiment,
    collect_experiments,
    component_runtime_key,
    component_venv_root,
    decomposer_dataset,
    gym_venv,
    models_for_experiment,
    preparation_manifest,
    source_dataset,
)

AGENT_REF = {"type": "responses_api_agents", "name": "decomposer"}
SFT_SPECS = {
    "workplace-all-v3": "workplace_all_v3.yaml",
    "workplace-26b-nonthinking-v3": "workplace_26b_nonthinking_v3.yaml",
    "workplace-deepseek-e4b-thinking-v1": "workplace_deepseek_e4b_thinking_v1.yaml",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def count_jsonl(path: Path) -> int:
    rows = 0
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            rows += 1
    return rows


def prepare_decomposer_dataset(source: Path, destination: Path, rows: int) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    actual_rows = 0
    with source.open() as input_stream, temporary.open("w") as output_stream:
        for line_number, line in enumerate(input_stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                temporary.unlink(missing_ok=True)
                raise TypeError(f"{source}:{line_number}: expected a JSON object")
            record["agent_ref"] = AGENT_REF
            output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            actual_rows += 1
    if actual_rows != rows:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Expected {rows} rows, found {actual_rows}")
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "rows": actual_rows,
        "sha256": sha256_file(destination),
        "agent_ref": AGENT_REF,
    }


def _uv_executable() -> Path:
    if UV_BIN.is_file() and os.access(UV_BIN, os.X_OK):
        return UV_BIN
    discovered = shutil.which("uv")
    if discovered:
        return Path(discovered)
    raise FileNotFoundError("uv is required to prepare Workplace Assistant runtimes")


def prepare_gym_runtime(repo_root: Path, *, create: bool) -> dict[str, Any]:
    uv = _uv_executable()
    venv = gym_venv(repo_root)
    gym_bin = venv / "bin" / "gym"
    if not gym_bin.is_file() and create:
        venv.parent.mkdir(parents=True, exist_ok=True)
        UV_CACHE.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": str(venv),
            "UV_CACHE_DIR": str(UV_CACHE),
            "UV_LINK_MODE": "copy",
        }
        subprocess.run(
            [str(uv), "sync", "--project", str(repo_root / "external" / "Gym"), "--frozen"],
            check=True,
            env=env,
        )
    if not gym_bin.is_file():
        raise FileNotFoundError(f"Prepared Gym executable is missing: {gym_bin}")
    lock = repo_root / "external" / "Gym" / "uv.lock"
    return {
        "path": str(venv),
        "gym_bin": str(gym_bin),
        "python": str(venv / "bin" / "python"),
        "uv_lock_sha256": sha256_file(lock),
        "uv_bin": str(uv),
        "uv_sha256": sha256_file(uv),
    }


def _runtime_versions(gym_python: Path) -> dict[str, str]:
    lines = subprocess.run(
        [
            str(gym_python),
            "-c",
            "import importlib.metadata as m; print(m.version('ray')); print(m.version('openai'))",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"ray": lines[0], "openai": lines[1]}


def components_for_experiments(experiments: Sequence[Experiment]) -> tuple[str, ...]:
    components = {"resources_servers/workplace_assistant"}
    if any(experiment.kind == "simple" for experiment in experiments):
        components.update(
            {
                "responses_api_agents/simple_agent",
                "responses_api_models/vllm_model",
            }
        )
    if any(experiment.kind == "decomposer" for experiment in experiments):
        components.update(
            {
                "responses_api_agents/decomposer_agent",
                "responses_api_models/openai_model",
            }
        )
    return tuple(sorted(components))


def prepare_component_venvs(
    repo_root: Path,
    components: Sequence[str],
    versions: dict[str, str],
    *,
    create: bool,
) -> dict[str, Any]:
    uv = _uv_executable()
    root = component_venv_root(repo_root)
    env = {**os.environ, "UV_CACHE_DIR": str(UV_CACHE), "UV_LINK_MODE": "copy"}
    prepared: dict[str, Any] = {}
    for component in components:
        component_dir = repo_root / "external" / "Gym" / component
        venv = root / component / ".venv"
        python = venv / "bin" / "python"
        activate = venv / "bin" / "activate"
        if not (python.is_file() and activate.is_file()) and create:
            venv.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    str(uv),
                    "venv",
                    "--seed",
                    "--allow-existing",
                    "--python",
                    "3.12",
                    str(venv),
                ],
                check=True,
                env=env,
            )
        if not (python.is_file() and activate.is_file()):
            raise FileNotFoundError(f"Prepared component environment is missing: {venv}")

        version_command = [
            str(python),
            "-c",
            "import importlib.metadata as m; print(m.version('ray')); print(m.version('openai'))",
        ]
        probe = subprocess.run(
            version_command,
            check=False,
            capture_output=True,
            text=True,
        )
        installed = probe.stdout.splitlines() if probe.returncode == 0 else []
        if installed != [versions["ray"], versions["openai"]] and create:
            requirements = component_dir / "requirements.txt"
            if requirements.is_file():
                install_target = ["-r", str(requirements)]
            elif (component_dir / "pyproject.toml").is_file():
                # Component pyprojects use a uv-only relative source for
                # `nemo-gym[dev]`. Install that source explicitly so `uv pip`
                # never resolves an unrelated index package and does not need
                # to write a component-local lockfile.
                install_target = [
                    "-e",
                    f"{repo_root / 'external' / 'Gym'}[dev]",
                ]
            else:
                raise FileNotFoundError(
                    f"Component has no requirements or pyproject: {component_dir}"
                )
            subprocess.run(
                [str(uv), "pip", "install", "--python", str(python), *install_target],
                check=True,
                cwd=component_dir,
                env=env,
            )
            subprocess.run(
                [
                    str(uv),
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    f"ray[default]=={versions['ray']}",
                    f"openai=={versions['openai']}",
                ],
                check=True,
                env=env,
            )
        installed = subprocess.run(
            version_command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if installed != [versions["ray"], versions["openai"]]:
            raise RuntimeError(f"Runtime version mismatch in {venv}: {installed}")
        prepared[component] = {"venv": str(venv), "python": str(python)}
    return {
        "root": str(root),
        "key": component_runtime_key(repo_root),
        "components": prepared,
        "versions": versions,
    }


def validate_checkpoint(checkpoint: Path, *, full_hashes: bool = False) -> dict[str, Any]:
    required_metadata = ("config.json", "tokenizer_config.json")
    missing = [name for name in required_metadata if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{checkpoint}: missing {', '.join(missing)}")
    index_path = checkpoint / "model.safetensors.index.json"
    monolithic_path = checkpoint / "model.safetensors"
    required_hashes = set(required_metadata)
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            raise ValueError(f"{index_path}: weight_map has no shards")
        required_hashes.add(index_path.name)
    elif monolithic_path.is_file():
        shards = [monolithic_path.name]
    else:
        raise FileNotFoundError(
            f"{checkpoint}: missing model.safetensors or model.safetensors.index.json"
        )
    for shard in shards:
        path = checkpoint / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{checkpoint}: missing or empty shard {shard}")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in checkpoint.iterdir() if item.is_file()):
        entry: dict[str, Any] = {"size": path.stat().st_size}
        if full_hashes or path.name in required_hashes:
            entry["sha256"] = sha256_file(path)
        files[path.name] = entry
    return {
        "path": str(checkpoint),
        "safetensors_shards": shards,
        "files": files,
    }


def _experiment_models(
    experiment: Experiment, *, full_hashes: bool
) -> dict[str, Any]:
    if isinstance(experiment, SimpleExperiment):
        return {
            "policy": validate_checkpoint(
                experiment.checkpoint, full_hashes=full_hashes
            )
        }
    return {
        "subagents": {
            model.model_id: validate_checkpoint(
                model.snapshot, full_hashes=full_hashes
            )
            for model in models_for_experiment(experiment)
        }
    }


def _run_upstream_preparer(
    repo_root: Path, gym_python: Path, split: str, destination_dir: Path
) -> Path:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (str(repo_root), str(repo_root / "external" / "Gym"))
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    subprocess.run(
        [
            str(gym_python),
            "-m",
            "gyms.workplace_assistant.prepare",
            "_upstream",
            "--repo-root",
            str(repo_root),
            "--split",
            split,
            "--output-dir",
            str(destination_dir),
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )
    return destination_dir / f"{split}.jsonl"


def prepare_source(
    repo_root: Path,
    gym_python: Path,
    split: str,
    *,
    reuse_source: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_rows = SPLIT_ROWS[split]
    raw_path = source_dataset(split)
    if not reuse_source or not raw_path.is_file():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"workplace-{split}-") as temporary:
            prepared = _run_upstream_preparer(
                repo_root, gym_python, split, Path(temporary)
            )
            if count_jsonl(prepared) != expected_rows:
                raise ValueError(
                    f"Upstream {split} split does not contain {expected_rows} rows"
                )
            atomic_copy(prepared, raw_path)
    rows = count_jsonl(raw_path)
    if rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} {split} rows, found {rows}")
    raw_manifest = {
        "path": str(raw_path),
        "split": split,
        "rows": rows,
        "sha256": sha256_file(raw_path),
    }
    decomposer_manifest = prepare_decomposer_dataset(
        raw_path, decomposer_dataset(split), rows
    )
    decomposer_manifest["split"] = split
    return raw_manifest, decomposer_manifest


def prepare_eval(args: argparse.Namespace) -> int:
    repo_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    experiments = collect_experiments(
        tuple(args.experiment or ()), tuple(args.filter or ())
    )
    if not experiments:
        raise ValueError("Experiment selectors matched no registered experiments")
    runtime = prepare_gym_runtime(repo_root, create=not args.skip_runtime)
    gym_python = Path(runtime["python"])
    versions = _runtime_versions(gym_python)
    components = components_for_experiments(experiments)
    component_runtime = prepare_component_venvs(
        repo_root, components, versions, create=not args.skip_runtime
    )
    raw_dataset, prepared_dataset = prepare_source(
        repo_root,
        gym_python,
        args.split,
        reuse_source=args.reuse_source,
    )
    project_tools = {"vllm": PROJECT_VENV / "bin" / "vllm"}
    if any(experiment.kind == "decomposer" for experiment in experiments):
        project_tools["langgraph"] = PROJECT_VENV / "bin" / "langgraph"
    for name, path in project_tools.items():
        if not path.is_file():
            raise FileNotFoundError(f"Project {name} executable is missing: {path}")
    gym_commit = subprocess.run(
        ["git", "-C", str(repo_root / "external" / "Gym"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    summaries: list[dict[str, Any]] = []
    for experiment in experiments:
        experiment_components = components_for_experiments((experiment,))
        experiment_component_runtime = {
            **component_runtime,
            "components": {
                name: component_runtime["components"][name]
                for name in experiment_components
            },
        }
        manifest = {
            "schema_version": 2,
            "created_at": datetime.now(UTC).isoformat(),
            "experiment": {"name": experiment.name, "kind": experiment.kind},
            "split": args.split,
            "datasets": {"simple": raw_dataset, "decomposer": prepared_dataset},
            "gym": {"commit": gym_commit, "runtime": runtime},
            "component_runtime": experiment_component_runtime,
            "project_venv": str(PROJECT_VENV),
            "project_tools": {name: str(path) for name, path in project_tools.items()},
            "models": _experiment_models(
                experiment, full_hashes=args.full_checkpoint_hashes
            ),
        }
        path = preparation_manifest(args.split, experiment.name)
        atomic_json(path, manifest)
        summaries.append(
            {
                "experiment": experiment.name,
                "kind": experiment.kind,
                "manifest": str(path),
            }
        )
    print(json.dumps({"split": args.split, "prepared": summaries}, indent=2))
    return 0


def prepare_sft(args: argparse.Namespace) -> int:
    from data.sft import prepare_dataset

    spec = Path(__file__).with_name("sft_specs") / SFT_SPECS[args.dataset]
    prepared = prepare_dataset(spec, args.output_root)
    print(
        json.dumps(
            {
                "dataset": prepared.manifest["dataset"],
                "release_dir": str(prepared.release_dir),
                "manifest_path": str(prepared.manifest_path),
                "filtering": prepared.manifest["filtering"],
                "records": prepared.manifest["records"],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_upstream(args: argparse.Namespace) -> int:
    external = args.repo_root / "external" / "Gym"
    sys.path.insert(0, str(external))
    from environments.workplace_assistant import prepare as upstream

    args.output_dir.mkdir(parents=True, exist_ok=True)
    upstream.DATA_DIR = args.output_dir
    output = upstream.prepare(args.split)
    if output != args.output_dir / f"{args.split}.jsonl":
        raise ValueError(f"Unexpected upstream output path: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="Prepare evaluation inputs")
    eval_parser.add_argument("--split", choices=SPLITS, default="train")
    eval_parser.add_argument("--experiment", action="append")
    eval_parser.add_argument("--filter", action="append")
    eval_parser.add_argument("--skip-runtime", action="store_true")
    eval_parser.add_argument("--reuse-source", action="store_true")
    eval_parser.add_argument("--full-checkpoint-hashes", action="store_true")

    sft_parser = subparsers.add_parser("sft", help="Build a canonical SFT release")
    sft_parser.add_argument("--dataset", choices=tuple(SFT_SPECS), required=True)
    sft_parser.add_argument("--output-root", type=Path, default=SFT_OUTPUT_ROOT)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["_upstream"]:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--repo-root", type=Path, required=True)
        parser.add_argument("--split", choices=SPLITS, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        return run_upstream(parser.parse_args(arguments[1:]))
    args = build_parser().parse_args(arguments)
    if args.command == "eval":
        return prepare_eval(args)
    if args.command == "sft":
        return prepare_sft(args)
    raise AssertionError(f"Unhandled preparation command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
