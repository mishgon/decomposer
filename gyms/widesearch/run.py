"""Run WideSearch with live web tools and save answers for offline scoring."""
import asyncio
import os
from pathlib import Path

from gyms.browsecomp import run as browsecomp
from gyms.widesearch.dataset import TASKS

AGENTS = ["decomposer", "researcher_4b_thinking", "researcher_27b_thinking",
          "decomposer_4b_non_thinking", "decomposer_sft_1_2_0", "decomposer_sft_1_4_0"]
FINAL_FORMAT = "\n\nReturn the requested table in one fenced block labeled markdown (```markdown). Preserve the requested column names."
BACKEND = "keenable-pro-web-v1"


async def preflight(backend):
    if backend != BACKEND:
        raise ValueError(f"WideSearch requires {BACKEND}; runs from other backends cannot resume here")
    if not os.environ.get("KEENABLE_API_KEY"):
        raise ValueError("KEENABLE_API_KEY is required for WideSearch")


async def save_answer(task, output, root):
    (root / "answer.md").write_text(output)
    return {"status": "pending", "score": None, "passed": False}


async def main(args):
    if args.agent not in AGENTS:
        raise ValueError(f"Choose one of {AGENTS}")
    await preflight(args.retrieval_url)
    if args.episode:
        await browsecomp.episode(args, server_config=Path(__file__).with_name("langgraph.json"),
                                 final_format=FINAL_FORMAT, evaluator=save_answer)
    else:
        return await browsecomp.main(args, episode_module="gyms.widesearch.run", preflight_check=preflight)


def parser():
    cli = browsecomp.parser()
    cli.description = __doc__
    cli.set_defaults(data=TASKS, output_dir=browsecomp.REPO_ROOT / "artifacts/evals/widesearch",
                     agent="decomposer_sft_1_2_0", retrieval_url=BACKEND)
    return cli


if __name__ == "__main__":
    asyncio.run(main(parser().parse_args()))
