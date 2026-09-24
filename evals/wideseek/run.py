"""Run a fixed WideSeek evaluation and aggregate its native episode scores."""
import json
import statistics

from gyms.wideseek.run import cli, main as run_episodes
from gyms.wideseek.runtime import save


def summarize(rows):
    scores = [r["evaluation"]["score"] for r in rows if r["evaluation"]["score"] is not None]
    seconds = [r["agent_finished_at"] - r["started_at"] for r in rows]
    tokens = {field: sum(role[field] for r in rows for name, role in r["usage"].items() if name != "judge")
              for field in ("input_tokens", "output_tokens")}
    return {"attempts": len(rows), "scored": len(scores), "unscored": len(rows) - len(scores),
            "mean_native_score": statistics.mean(scores) if scores else None,
            "mean_native_score_infra_zero": sum(scores) / len(rows) if rows else None,
            "metrics": sorted({r["evaluation"].get("metric", "unscored") for r in rows}),
            "mean_agent_seconds": statistics.mean(seconds) if seconds else None,
            "median_agent_seconds": statistics.median(seconds) if seconds else None,
            "agent_tokens": tokens,
            "normal_finishes": sum(r["status"] == "finished" for r in rows)}


async def main(args):
    await run_episodes(args)
    rows = [json.loads(path.read_text())
            for path in (args.output / args.mode).glob("*/attempt-???/result.json")]
    save(args.output / f"{args.mode}-summary.json", summarize(rows))


if __name__ == "__main__":
    cli(main)
