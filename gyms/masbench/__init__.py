"""MASBench task data and isolated agent episodes."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = "Salesforce/MASBench"
REVISION = "f1d57e20c304c5e42fa77c1a0412de2bc7ad52a3"
AXES = ("depth", "horizon", "breadth", "parallel", "robustness")

RULES = """MASBench rules:
All arithmetic calculations are performed modulo 23. Any parameter not mentioned
in the problem statement defaults to zero. For magic-number retrieval questions,
return the exact number from the passage; do not reduce that number modulo 23.
Follow the task's requested final-answer format and preserve the problem order.
"""


def task_input(task):
    return {"messages": [{"role": "user", "content": RULES + "\n" + task["question"]}]}
