"""Read-only integrity audit for every public GAIA2 tool schema."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from gyms.gaia2.experiments import (
    DEFAULT_GAIA2_REPO,
    filesystem_dir,
    gaia2_venv,
    scenario_dir,
)

# ARE reads its filesystem source at import time. Point it at the immutable local
# snapshot before importing the submodule, and prefer the checked-out source over
# any older ARE installation in the caller's environment.
sys.path.insert(0, str(DEFAULT_GAIA2_REPO))
for _site_packages in gaia2_venv(DEFAULT_GAIA2_REPO).glob("lib/python*/site-packages"):
    sys.path.append(str(_site_packages))
os.environ.setdefault("DEMO_FS_PATH", str(filesystem_dir()))

from are.simulation.agents.default_agent.tools import native_tools  # noqa: E402
from are.simulation.apps import ALL_APPS  # noqa: E402
from are.simulation.environment import Environment, EnvironmentConfig  # noqa: E402
from are.simulation.scenarios.utils.load_utils import load_scenario  # noqa: E402
from are.simulation.tool_utils import AppTool, AppToolAdapter  # noqa: E402
from jsonschema import Draft202012Validator  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

from gyms.gaia2.broker import HIDDEN_AUI_TOOLS, app_tool_schema  # noqa: E402
from gyms.gaia2.subagents.graphs import _tool_from_schema  # noqa: E402

VALID_JSON_SCHEMA_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)
SIMPLE_HIDDEN_AUI_TOOLS = HIDDEN_AUI_TOOLS - {
    "AgentUserInterface__send_message_to_user"
}
EXPECTED_SIMPLE_ONLY_TOOLS = {"AgentUserInterface__send_message_to_user"}
BROAD_OBJECT_ALLOWLIST: frozenset[str] = frozenset()
REPRESENTATIVE_SCENARIOS = {
    "execution": "scenario_universe_25_vetd7u.json",
    "search": "scenario_universe_28_4sn4lc.json",
    "ambiguity": "scenario_universe_23_iy0pbp.json",
}


def _public_name(tool: AppTool) -> str:
    return tool._public_name or tool.name


def _ordered_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _walk_schema(value: Any, path: str = "$"):
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_schema(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_schema(item, f"{path}[{index}]")


def _inspect_schema(
    identity: str,
    tool: AppTool,
    schema: dict[str, Any],
    issues: dict[str, list[str]],
) -> None:
    parameters = schema["function"]["parameters"]
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    try:
        Draft202012Validator.check_schema(parameters)
    except Exception as error:
        issues["invalid_json_schemas"].append(f"{identity}: {error}")

    signature = inspect.signature(tool.function) if tool.function else None
    for argument in tool.args:
        argument_identity = f"{identity}.{argument.name}"
        parameter = (
            signature.parameters.get(argument.name) if signature is not None else None
        )
        if parameter is not None and parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            if argument.name in properties:
                issues["exposed_variadics"].append(argument_identity)
            continue
        if argument.has_default and argument.name in required:
            issues["defaulted_required"].append(argument_identity)
        argument_schema = properties.get(argument.name)
        if argument_schema == {} and argument_identity not in BROAD_OBJECT_ALLOWLIST:
            issues["broad_object_parameters"].append(argument_identity)
        if (
            isinstance(argument_schema, dict)
            and argument_schema.get("type") == "object"
            and argument_schema.get("additionalProperties") is not False
            and argument_identity not in BROAD_OBJECT_ALLOWLIST
        ):
            issues["broad_object_parameters"].append(argument_identity)

    for path, node in _walk_schema(parameters):
        if not isinstance(node, dict) or "type" not in node:
            continue
        declared = node["type"]
        values = declared if isinstance(declared, list) else [declared]
        invalid = [value for value in values if value not in VALID_JSON_SCHEMA_TYPES]
        if invalid:
            issues["invalid_type_names"].append(f"{identity}:{path}: {invalid!r}")
    for forbidden in ("args", "kwargs"):
        if forbidden in properties:
            issues["exposed_variadics"].append(f"{identity}.{forbidden}")


def _registry_audit(issues: dict[str, list[str]]) -> dict[str, Any]:
    schemas: list[dict[str, Any]] = []
    tool_count = 0
    python_argument_count = 0
    public_argument_count = 0
    for app_class in ALL_APPS:
        app = app_class()
        environment = Environment(EnvironmentConfig(duration=1))
        environment.register_apps([app])
        for tool in app.get_tools():
            identity = f"{app_class.__name__}:{_public_name(tool)}"
            tool_count += 1
            python_argument_count += len(tool.args)
            try:
                schema = app_tool_schema(tool)
            except Exception as error:
                issues["unsupported_annotations"].append(f"{identity}: {error}")
                continue
            public_argument_count += len(
                schema["function"]["parameters"].get("properties", {})
            )
            _inspect_schema(identity, tool, schema, issues)
            schemas.append(schema)
        del environment, app
    gc.collect()
    return {
        "app_count": len(ALL_APPS),
        "tool_count": tool_count,
        "python_argument_count": python_argument_count,
        "public_argument_count": public_argument_count,
        "schemas": schemas,
    }


def _representative_audit(issues: dict[str, list[str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for domain, filename in REPRESENTATIVE_SCENARIOS.items():
        path = scenario_dir(domain) / filename
        if not path.is_file():
            issues["missing_representative_data"].append(str(path))
            continue
        with tempfile.TemporaryDirectory(prefix=f"gaia2-audit-{domain}-") as sandbox:
            scenario = load_scenario(str(path))
            scenario.initialize(sandbox_dir=sandbox)
            registered = scenario.get_tools()
            simple_tools = [
                tool
                for tool in registered
                if _public_name(tool) not in SIMPLE_HIDDEN_AUI_TOOLS
                and tool.name not in SIMPLE_HIDDEN_AUI_TOOLS
            ]
            decomposer_tools = [
                tool
                for tool in registered
                if _public_name(tool) not in HIDDEN_AUI_TOOLS
                and tool.name not in HIDDEN_AUI_TOOLS
            ]

            adapters = [AppToolAdapter(tool) for tool in simple_tools]
            original_fallback = native_tools._schema_from_inputs

            def reject_fallback(tool):
                raise AssertionError(f"lossy native fallback used for {tool.name}")

            native_tools._schema_from_inputs = reject_fallback
            try:
                native_schemas = native_tools.build_openai_tools(adapters)
            except Exception as error:
                issues["native_fallback_or_conversion"].append(f"{domain}: {error}")
                native_schemas = []
            finally:
                native_tools._schema_from_inputs = original_fallback

            broker_schemas = [app_tool_schema(tool) for tool in decomposer_tools]
            native_by_name = {
                schema["function"]["name"]: schema for schema in native_schemas
            }
            broker_by_name = {
                schema["function"]["name"]: schema for schema in broker_schemas
            }
            if len(native_by_name) != len(native_schemas):
                issues["duplicate_tool_names"].append(f"{domain}: simple")
            if len(broker_by_name) != len(broker_schemas):
                issues["duplicate_tool_names"].append(f"{domain}: decomposer")
            native_names = list(native_by_name)
            broker_names = list(broker_by_name)
            common_names = [name for name in native_names if name in broker_by_name]
            if common_names != broker_names:
                issues["ordering_differences"].append(domain)
            simple_only = set(native_by_name) - set(broker_by_name)
            if simple_only != EXPECTED_SIMPLE_ONLY_TOOLS:
                issues["surface_differences"].append(
                    f"{domain}: unexpected simple-only tools {sorted(simple_only)!r}"
                )
            leaked = set(broker_by_name) & HIDDEN_AUI_TOOLS
            if leaked:
                issues["hidden_tool_leaks"].append(f"{domain}: {sorted(leaked)!r}")

            context = {
                "tool_schemas": broker_schemas,
                "broker_url": "http://audit.invalid/v1/sessions/audit",
                "session_token": "audit",
                "policy": "shared_serialized",
                "scenario_id": scenario.scenario_id,
                "run_number": None,
                "notification_cursor": 0,
            }
            for name in common_names:
                native_schema = native_by_name[name]
                broker_schema = broker_by_name[name]
                if _ordered_json(native_schema) != _ordered_json(broker_schema):
                    issues["native_broker_differences"].append(f"{domain}:{name}")
                langchain_schema = convert_to_openai_tool(
                    _tool_from_schema(broker_schema, context)
                )
                if _ordered_json(langchain_schema) != _ordered_json(broker_schema):
                    issues["langchain_broker_differences"].append(f"{domain}:{name}")

            result[domain] = {
                "scenario": path.name,
                "registered_tools": len(registered),
                "simple_tools": len(native_schemas),
                "decomposer_tools": len(broker_schemas),
                "common_tools": len(common_names),
                "simple_only_tools": sorted(simple_only),
                "schemas": {
                    "native": native_schemas,
                    "broker": broker_schemas,
                },
            }
            del scenario
            gc.collect()
    return result


def run_audit() -> dict[str, Any]:
    issue_names = (
        "broad_object_parameters",
        "defaulted_required",
        "duplicate_tool_names",
        "exposed_variadics",
        "hidden_tool_leaks",
        "invalid_json_schemas",
        "invalid_type_names",
        "langchain_broker_differences",
        "missing_representative_data",
        "native_broker_differences",
        "native_fallback_or_conversion",
        "ordering_differences",
        "surface_differences",
        "unsupported_annotations",
    )
    issues = {name: [] for name in issue_names}
    registry = _registry_audit(issues)
    representative = _representative_audit(issues)
    checksum_payload = {
        "registry": registry["schemas"],
        "representative": {
            domain: item["schemas"] for domain, item in representative.items()
        },
    }
    checksum = hashlib.sha256(
        json.dumps(
            checksum_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    registry = {key: value for key, value in registry.items() if key != "schemas"}
    representative = {
        domain: {key: value for key, value in item.items() if key != "schemas"}
        for domain, item in representative.items()
    }
    return {
        "registry": registry,
        "representative_scenarios": representative,
        "issues": issues,
        "schema_checksum": checksum,
        "ok": not any(issues.values()),
    }


def _seeded_audits() -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for seed in ("0", "1", "42"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-m", "gyms.gaia2.audit_tools", "--single-seed"],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        marker = "GAIA2_TOOL_AUDIT="
        lines = [
            line for line in completed.stdout.splitlines() if line.startswith(marker)
        ]
        if not lines:
            raise RuntimeError(
                f"GAIA2 tool audit failed under PYTHONHASHSEED={seed}:\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        reports[seed] = json.loads(lines[-1][len(marker) :])
    checksums = {report["schema_checksum"] for report in reports.values()}
    report = reports["0"]
    report["hash_seed_checksums"] = {
        seed: item["schema_checksum"] for seed, item in reports.items()
    }
    if len(checksums) != 1:
        report["issues"]["nondeterministic_checksums"] = sorted(checksums)
        report["ok"] = False
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-seed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    report = run_audit() if args.single_seed else _seeded_audits()
    if args.single_seed:
        print("GAIA2_TOOL_AUDIT=" + json.dumps(report, separators=(",", ":")))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
