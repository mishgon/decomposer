"""DeepDive QA adapted to WideSeek's offline Wikipedia environment."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "artifacts/gyms/deepdive/data/qa_rl/tasks.jsonl"
