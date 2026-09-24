"""The repository layout: which top-level packages may import which.

`gyms/<gym>` runs agents and saves raw results; `evals`, `sft`, `opd` and `rl` build
on it, never the other way round. `evals` stays independent of training code, and
`sft` and `opd` do not reach into evaluations.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_IMPORTS = {
    "gyms": {"evals", "sft", "opd", "rl"},
    "sft": {"evals", "opd", "rl"},
    "evals": {"sft", "opd", "rl"},
    "opd": {"evals", "rl"},
}


def _imported_top_levels(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("package", sorted(FORBIDDEN_IMPORTS))
def test_packages_only_import_downstream_layers(package):
    violations = [
        f"{path.relative_to(REPO_ROOT)} imports {name}"
        for path in sorted((REPO_ROOT / package).rglob("*.py"))
        for name in sorted(_imported_top_levels(path) & FORBIDDEN_IMPORTS[package])
    ]
    assert not violations, "\n".join(violations)


def test_the_old_training_and_data_roots_are_gone():
    assert not (REPO_ROOT / "training").exists()
    assert not (REPO_ROOT / "data").exists()


def test_every_gym_has_its_layer_directories():
    for gym in ("workplace_assistant", "gaia2", "tau2_gym"):
        for layer in ("gyms", "evals", "sft", "opd", "rl"):
            assert (REPO_ROOT / layer / gym).is_dir(), f"{layer}/{gym}"
