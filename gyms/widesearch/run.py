"""Run WideSearch with BrowseComp agentic-rag tools and save answers for offline scoring."""
import asyncio
from pathlib import Path

from gyms.browsecomp import run as browsecomp
from gyms.widesearch.dataset import TASKS

AGENTS = ["decomposer", "researcher_4b_thinking", "researcher_27b_thinking",
          "decomposer_4b_non_thinking", "decomposer_sft_1_1_0", "decomposer_sft_1_2_0", "decomposer_sft_1_4_0", "decomposer_sft_1_5_0",
          "decomposer_sft_1_5_0_timur", "researcher_timur_non_thinking"]
FINAL_FORMAT = "\n\nReturn the requested table in one fenced block labeled markdown (```markdown). Preserve the requested column names."
BACKEND = "http://127.0.0.1:18082"


preflight = browsecomp.preflight


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
