"""Collection-only teacher; the reusable gym and evaluations have no SFT cap."""
import asyncio
from transformers import AutoTokenizer
from agents import decomposer as gym_decomposer
from sft.sequence_limit import StudentSequenceLimit


async def decomposer():
    tokenizer = await asyncio.to_thread(
        AutoTokenizer.from_pretrained, '/opt/sft-tokenizer', local_files_only=True)
    return gym_decomposer(middleware=[StudentSequenceLimit(tokenizer)])
