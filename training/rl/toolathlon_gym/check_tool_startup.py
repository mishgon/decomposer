"""Start real task tools without model calls; compare baked versus source launcher."""
import argparse
import json
from pathlib import Path
import time

from gyms.toolathlon_gym.episode import Episode


class BakedLauncherEpisode(Episode):
    def start_task_container(self, *args):
        args = list(args)
        for i in range(len(args) - 1, 0, -1):
            if args[i].endswith("/subagents/webapp.py:ro") and args[i - 1] == "-v":
                del args[i - 1:i + 1]
        return super().start_task_container(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="localhost/decomposer-toolathlon-rl:status-filter-fix")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path("external/toolathlon_gym/tasks/finalpool")
    pool = json.loads(Path("training/rl/toolathlon_gym/rl_task_pool.json").read_text())
    configs = {t["task_id"]: json.loads((root / t["task_id"] / "task_config.json").read_text())
               for t in pool["tasks"]}
    remaining = set().union(*(set(c["needed_mcp_servers"]) for c in configs.values()))
    selected = []
    while remaining:
        task = max(sorted(configs), key=lambda t: len(remaining & set(configs[t]["needed_mcp_servers"])))
        selected.append(task)
        remaining -= set(configs[task]["needed_mcp_servers"])
    rows = []
    for task in selected:
        for label, cls in ([("baked", BakedLauncherEpisode)] if args.compare else []) + [("fixed", Episode)]:
            start = time.monotonic()
            episode = cls(task, args.output / label / task, image=args.image)
            row = {"task": task, "launcher": label, "tools": configs[task]["needed_mcp_servers"]}
            try:
                episode.start()
                row["ready_seconds"] = time.monotonic() - start
                row["status"] = "ready"
            except Exception as error:
                row.update(status="error", error=repr(error))
            finally:
                episode.close()
            log = (episode.directory / f"{episode.container}.log").read_text()
            row["package_builds"] = log.count("Building ")
            row["package_installs"] = log.count("Installed ")
            row["cleanup_errors"] = json.loads((episode.directory / "cleanup.json").read_text())["errors"]
            rows.append(row)
            (args.output / "results.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
    if any(r["status"] != "ready" or r["package_builds"] or r["package_installs"] or r["cleanup_errors"]
           for r in rows if r["launcher"] == "fixed"):
        raise SystemExit(1)
