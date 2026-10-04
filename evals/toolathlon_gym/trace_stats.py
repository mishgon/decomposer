"""Descriptive raw trace statistics, without loading a model or tokenizer."""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def summarize_traces(root):
    totals = Counter()
    models = Counter()
    durations = []
    unreadable = []
    for path in sorted(Path(root).rglob("trace.json")):
        try:
            trace = json.loads(path.read_text())
            totals["traces"] += 1
            totals["agent_errors"] += bool(trace.get("agent_error"))
            totals["subagent_runs"] += len(trace.get("subagent_runs", {}))
            models[str(trace.get("decomposer_model") or trace.get("subagent_api_model"))] += 1
            if trace.get("started_at") and trace.get("finished_at"):
                durations.append((datetime.fromisoformat(trace["finished_at"]) -
                                  datetime.fromisoformat(trace["started_at"])).total_seconds())
            usage_path = path.with_name("usage.json")
            if usage_path.is_file():
                usage = json.loads(usage_path.read_text()).get("totals", {})
                for field in ("input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens", "model_responses"):
                    totals[field] += usage.get(field, 0)
        except (OSError, ValueError, TypeError) as error:
            unreadable.append({"path": str(path), "error": str(error)})
    return {"totals": dict(totals), "models": dict(models),
            "mean_episode_seconds": sum(durations) / len(durations) if durations else None,
            "timed_episodes": len(durations), "unreadable": unreadable}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_traces(args.root), indent=2))


if __name__ == "__main__":
    main()
