"""Remove only environments recorded by this throughput test, including interrupted setup."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from gyms.toolathlon_gym.episode import Episode


def close(path):
    prior = path.with_name("cleanup.json")
    if prior.exists() and not json.loads(prior.read_text()).get("errors"):
        return
    resources = json.loads(path.read_text())
    episode = object.__new__(Episode)
    episode.directory = path.parent
    for key in ("engine", "container", "pg", "network"):
        setattr(episode, key, resources[key])
    if not episode.network.startswith("decomposer-rl-"):
        raise ValueError("Unexpected environment ownership")
    episode.close()
    if json.loads(prior.read_text())["errors"]:
        raise RuntimeError(f"Cleanup failed: {prior}")


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(close, root.rglob("resources.json")))
