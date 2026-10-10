"""Submit GAIA2 evaluations to MLSpace; each job runs `evals.gaia2.run`.

Takes the flags of `gyms/gaia2/run_eval.py`, which stages the repo and builds the
jobs; only the script each job executes differs. Trace generation is submitted
with the gym launcher.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gyms.gaia2 import run_eval as launcher  # noqa: E402

ENTRYPOINT = ("evals", "gaia2", "run.py")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "trace-generation" in arguments:
        raise SystemExit("evals.gaia2.submit submits evaluations; use gyms.gaia2.run_eval for trace generation")
    return launcher.main(arguments, entrypoint=ENTRYPOINT)


if __name__ == "__main__":
    raise SystemExit(main())
