"""Integration check: native evaluators on empty output, without model calls."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gyms.toolathlon_gym.episode import Episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image", default="decomposer-toolathlon-rl:latest")
    args = parser.parse_args()
    manifest = json.loads(args.split.read_text())
    tasks = sorted(set(manifest["train"] + manifest["validation"]))
    args.output.mkdir(parents=True, exist_ok=False)

    def check(task):
        episode = Episode(task, args.output / task, subagent_port=8025, image=args.image)
        started = False
        try:
            episode.start()
            started = True
            result = {"task": task, "reward": episode.score()["reward"]}
        except Exception as error:
            result = {"task": task, "error": repr(error)}
        finally:
            if started:
                episode.close()
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(check, tasks))
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    if any("error" in result for result in results):
        raise SystemExit("Evaluator preflight failed; inspect saved native outputs")


if __name__ == "__main__":
    main()
