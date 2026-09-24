"""Evaluate a fixed task selection using the shared Gym executor."""

import json
import signal

from gyms.toolathlon_gym import run as gym
from gyms.toolathlon_gym.parallel import run
from .summary import summarize


def main(argv=None):
    parser = gym.create_parser()
    parser.set_defaults(purpose="evaluation", output_dir=gym.REPO_ROOT / "artifacts/evals/toolathlon_gym")
    parser.add_argument("--denominator", type=int)
    args = parser.parse_args(argv)
    denominator = args.denominator
    del args.denominator
    root = run(args)
    summary = summarize(root, denominator)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return root


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, gym._handle_termination)
    main()
