"""Submit Workplace Assistant evaluations to MLSpace; each job runs `evals.workplace_assistant.run`.

Takes the flags of `gyms/workplace_assistant/run_eval.py`, which stages the repo and
builds the jobs; only the script each job executes differs.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gyms.workplace_assistant import run_eval as launcher  # noqa: E402

ENTRYPOINT = ("evals", "workplace_assistant", "run.py")


def main(argv: Sequence[str] | None = None) -> int:
    return launcher.main(argv, entrypoint=ENTRYPOINT)


if __name__ == "__main__":
    raise SystemExit(main())
