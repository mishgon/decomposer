import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
TOOLATHLON_ROOT = PROJECT_ROOT / "external" / "toolathlon_gym"
PREPARE_TASK = """
import sys
from pathlib import Path

from gyms.toolathlon_gym.task import prepare_task

prepare_task(
    sys.argv[1],
    Path(sys.argv[2]),
    toolathlon_root=Path(sys.argv[3]),
    run_preprocess=False,
)
"""


def test_prepare_task_without_preprocess(tmp_path: Path) -> None:
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    (stub_dir / "termcolor.py").write_text(
        "def colored(text, *args, **kwargs):\n    return text\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(
                None,
                [
                    str(stub_dir),
                    str(PROJECT_ROOT),
                    os.environ.get("PYTHONPATH"),
                ],
            )
        ),
        "TOOLATHLON_ROOT": str(TOOLATHLON_ROOT),
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            PREPARE_TASK,
            "wc-shipping-analysis",
            str(output_dir),
            str(TOOLATHLON_ROOT),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    runtime = json.loads((output_dir / "runtime.json").read_text())
    task_config = runtime["task_config"]
    assert task_config["task_dir"] == "wc-shipping-analysis"
    assert task_config["agent_workspace"] == str(output_dir / "workspace")
    assert (output_dir / "workspace").is_dir()
