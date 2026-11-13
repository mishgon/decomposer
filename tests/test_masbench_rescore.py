import asyncio
import json
from pathlib import Path

import pytest

from evals.masbench.rescore import normalize_answer, rescore, upstream_comparator


@pytest.mark.parametrize("answer,count,expected", [
    ("**Answer: 0**\nExplanation: 3 * 12 = 36.", 1, r"\boxed{0}"),
    ("Answer: 9\nAnswer: 6", 1, None),
    ("Answer: 9 or 6", 1, None),
    ("Answer: 9.5", 1, None),
    ("Answer: 3 + 6", 1, None),
    ("Answer: 9\nAnswer: cannot determine", 1, None),
    ("The passage contains 9 and 6.", 1, None),
    ("**Final Answer:**\n9", 1, r"\boxed{9}"),
    ("Problem 1: **3**\nProblem 2: **20**", 2, "\\boxed{3}\n\\boxed{20}"),
    ("Problem 2: 20\nProblem 1: 3", 2, None),
    ("Problem 1: 3\nProblem 1: 20", 2, None),
    ("Problem 1: 3\nProblem 2: 20\nProblem 3: 7", 2, None),
    ("Problem 1: value is 3\nProblem 2: value is 20", 2, None),
    ("\\boxed{0}\n\\boxed{0}\n### Final Answers\nProblem 1: \\boxed{0}\nProblem 2: \\boxed{0}",
     2, "\\boxed{0}\n\\boxed{0}"),
    (r"\boxed{0}\boxed{0}\boxed{0}\boxed{0}", 2, None),
    ("Answer: 3\n### Final Answer\n8", 1, r"\boxed{8}"),
])
def test_explicit_answer_normalization(answer, count, expected):
    normalized, method = normalize_answer(answer, count)
    assert normalized == (answer if expected is None else expected)
    assert (method == "abstain") == (expected is None)


def test_upstream_requires_exact_source(tmp_path):
    source = tmp_path / "mas_r1_reasoner/rewards/utils"
    source.mkdir(parents=True)
    (source / "compute_score.py").write_text("different revision")
    with pytest.raises(ValueError, match="hash mismatch"):
        upstream_comparator(tmp_path)


def test_rescore_preserves_failures_and_reports_demotions(monkeypatch):
    async def compare(answer, gold):
        return answer == r"\boxed{" + gold + "}" or answer == "Answer: 2\n9"

    monkeypatch.setattr("evals.masbench.rescore.upstream_comparator", lambda source: compare)
    rows = [{"task_id": str(i), "axis": "depth", "value": "2", "agent": "single",
             "status": status, "directory": str(i),
             "evaluation": {"passed": False, "expected": ["9"]}}
            for i, status in enumerate(["finished", "timeout", "error"])]
    snapshot = {"manifest": {"episodes": rows},
                "answers": [{"directory": str(i), "answer": "Answer: 2\n9"} for i in range(3)]}
    before = json.dumps(snapshot)
    result = asyncio.run(rescore(snapshot, Path("unused")))
    assert json.dumps(snapshot) == before
    assert [row["official"] for row in result["episodes"]] == [True, False, False]
    assert not any(row["normalized"] for row in result["episodes"])
    assert result["groups"][0]["demoted"] == 1
