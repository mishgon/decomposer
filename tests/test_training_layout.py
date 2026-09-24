"""Folder moves must preserve launcher roots and the shared OPD recipe."""
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_opd_recipe_is_the_rl_recipe():
    assert (ROOT / "opd/toolathlon_gym/rl_recipe.yaml").resolve() == (
        ROOT / "rl/toolathlon_gym/full.yaml")


def test_launcher_roots_and_shell_syntax():
    for stage in ("rl", "opd"):
        for script in (ROOT / stage / "toolathlon_gym").glob("*.sh"):
            subprocess.run(["bash", "-n", str(script)], check=True)
            assert (script.parent / "../..").resolve() == ROOT
            assert "/../../.." not in script.read_text()


def test_gym_has_no_training_or_evaluation_workflow_imports():
    for path in (ROOT / "gyms/toolathlon_gym").rglob("*.py"):
        for line in path.read_text().splitlines():
            assert not line.strip().startswith(("from rl.", "from opd.", "from evals.", "from sft."))
