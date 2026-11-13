"""Audit saved answers with pinned upstream scoring and explicit-answer normalization."""
import argparse
import ast
from collections import defaultdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import runpy
import asyncio

from math_verify import parse, verify
from math_verify.errors import TimeoutException

from gyms.masbench.evaluate import boxed_answers


UPSTREAM_REVISION = "0527c70a508eb49ed42d5dd49a91ea857551e242"
SOURCE_HASHES = {
    "compute_score.py": "8c6798115730b9063a43d57c39e3e61e5663ee9590e811c5bbdca162b5332312",
    "string_match_score.py": "371753dc5bfd9498e25b235f570e3185d20c7755daa81d4c1c275f1afffd0a71",
}


def upstream_comparator(source):
    """Load the original comparison functions without the VERL training runtime."""
    source = source / "mas_r1_reasoner/rewards/utils"
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Upstream source hash mismatch: {name}")
    scorer = runpy.run_path(str(source / "string_match_score.py"))["MathScorer"]()
    names = {"_fix_leading_zero_numbers", "_extract_all_boxed_answers",
             "_compare_multi_answers", "_compare_answers"}
    tree = ast.parse((source / "compute_score.py").read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    namespace = {"re": re, "parse": parse, "verify": verify,
                 "TimeoutException": TimeoutException, "Dict": dict, "List": list,
                 "Any": object, "main_rank_print": lambda *args: None,
                 "get_global": lambda name: False if name == "global_use_llm_judge" else ""}
    exec(compile(tree, str(source / "compute_score.py"), "exec"), namespace)

    async def compare(answer, gold):
        return bool(await namespace["_compare_answers"](
            "", answer, gold, {"math_scorer": scorer}, True, "<<horizon>>" in gold,
        ))

    return compare


def integer(text):
    """Accept a whole integer field, never a number embedded in an explanation."""
    text = text.strip().replace("**", "").replace("`", "").strip()
    for left, right in ((r"\(", r"\)"), ("$", "$")):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left):-len(right)].strip()
    boxes = boxed_answers(text)
    if len(boxes) == 1 and text == r"\boxed{" + boxes[0] + "}":
        text = boxes[0]
    if re.fullmatch(r"[+-]?\d+\.?", text):
        return str(int(text.rstrip(".")))
    return None


def normalize_answer(text, count):
    """Normalize only explicit integer answers; never inspect reference values.

    A last Final Answer(s) heading takes precedence over preceding discussion.
    Otherwise conflicting labels or duplicate boxes cause abstention. Ordered
    multi-answer labels must cover exactly 1..count. No arithmetic is evaluated.
    """
    headings = list(re.finditer(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?Final Answers?(?:\*\*)?\s*:?\s*$", text,
    ))
    section = text[headings[-1].end():] if headings else text
    boxes = boxed_answers(section)
    if boxes:
        values = [integer(value) for value in boxes]
        if len(values) == count and all(value is not None for value in values):
            return "\n".join(r"\boxed{" + value + "}" for value in values), "boxes"
        return text, "abstain"
    clean = section.replace("**", "")
    if count == 1:
        direct = integer(clean)
        if direct is not None:
            return r"\boxed{" + direct + "}", "integer"
        fields = re.findall(r"(?im)^\s*(?:#{1,6}\s*)?(?:Final\s+)?Answer\s*:\s*([^\n]+)", clean)
        values = [integer(field) for field in fields]
        if values and None not in values and len(set(values)) == 1:
            return r"\boxed{" + values[0] + "}", "answer_label"
    else:
        fields = re.findall(
            r"(?im)^\s*(?:#{1,6}\s*)?(?:Problem|Answer|P|Task)\s*(\d+)\s*(?:Answer)?\s*:\s*([^\n]+)",
            clean,
        )
        indices = [int(index) for index, _ in fields]
        values = [integer(field) for _, field in fields]
        if indices == list(range(1, count + 1)) and None not in values:
            return "\n".join(r"\boxed{" + value + "}" for value in values), "ordered_labels"
    return text, "abstain"


async def rescore(snapshot, source):
    compare = upstream_comparator(source)
    answers = {row["directory"]: row["answer"] for row in snapshot["answers"]}
    episodes = []
    groups = defaultdict(list)
    for row in snapshot["manifest"]["episodes"]:
        answer = answers[row["directory"]]
        expected = row["evaluation"]["expected"]
        gold = "<<horizon>>".join(expected)
        normalized, method = normalize_answer(answer, len(expected))
        official = corrected = False
        if row["status"] == "finished":
            official = await compare(answer, gold)
            corrected = official if normalized == answer else await compare(normalized, gold)
        record = {key: row[key] for key in ("task_id", "axis", "value", "agent", "status", "directory")}
        record.update(original=row["evaluation"]["passed"], official=official,
                      normalized=corrected, normalization=method,
                      normalized_answer=normalized if normalized != answer else None,
                      answer_sha256=hashlib.sha256(answer.encode()).hexdigest())
        episodes.append(record)
        for axis in (row["axis"], "all"):
            groups[(row["agent"], axis)].append(record)
    summary = []
    for (agent, axis), rows in sorted(groups.items()):
        summary.append({"agent": agent, "axis": axis, "n": len(rows),
                        **{metric: sum(row[metric] for row in rows)
                           for metric in ("original", "official", "normalized")},
                        "promoted": sum(not row["official"] and row["normalized"] for row in rows),
                        "demoted": sum(row["official"] and not row["normalized"] for row in rows)})
    return {"upstream_revision": UPSTREAM_REVISION, "source_hashes": SOURCE_HASHES,
            "protocol": "upstream deterministic answer correctness, LLM judge disabled",
            "normalized_protocol": "explicit integer fields or last Final Answers section; ordered exact match",
            "groups": summary, "episodes": episodes}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("snapshot", type=Path)
    cli.add_argument("--upstream", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    raw = args.snapshot.read_bytes()
    result = asyncio.run(rescore(json.loads(raw), args.upstream))
    result["packages"] = {}
    for package in ("math-verify", "sympy", "pylatexenc"):
        try:
            result["packages"][package] = version(package)
        except PackageNotFoundError:
            result["packages"][package] = None
    result["snapshot_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
