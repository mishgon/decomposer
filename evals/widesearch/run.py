"""Score saved WideSearch answers with the pinned upstream evaluator."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4
from unittest.mock import patch

import pandas as pd

from decomposer.models import create_model
from gyms.infrastructure import classify
from gyms.browsecomp import REPO_ROOT
from gyms.browsecomp.agents import JUDGE_MODEL
from gyms.browsecomp.run import save
from gyms.widesearch.dataset import SOURCE, REVISION

UPSTREAM = REPO_ROOT / "external/WideSearch"
sys.path.insert(0, str(UPSTREAM))
from src.evaluation.data_loader import WideSearchQuery, WideSearchResponse
from src.evaluation.evaluation import evaluate_single_query
from src.evaluation import metric_utils
from src.utils.utils import norm_column


def load_queries(source):
    queries = {}
    for line in (source / "widesearch.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["language"] != "en":
            continue
        evaluation = row["evaluation"]
        if isinstance(evaluation, str):
            evaluation = json.loads(evaluation)
        answer = pd.read_csv(source / "widesearch_gold" / f"{row['instance_id']}.csv")
        answer.columns = [norm_column(col) for col in answer.columns]
        queries[row["instance_id"]] = WideSearchQuery(
            row["instance_id"], row["query"], evaluation,
            answer[evaluation["required"]], row["language"])
    return queries


def score(query, answer, directory):
    model = create_model(JUDGE_MODEL)
    calls = []

    def completion(messages, model_config_name, *, expected_keys=None):
        response = model.invoke(messages)
        calls.append({"model": JUDGE_MODEL, "prompt": messages,
                      "response": response.model_dump(mode="json")})
        save(directory / "judge_calls.json", calls)
        parsed = metric_utils.parse_markdown_json(response.content)
        if not isinstance(parsed, dict):
            raise ValueError("Judge returned no JSON object")
        if expected_keys is not None:
            if set(parsed) != expected_keys or any(type(v) not in (int, float) or v not in (0, 1)
                                                  for v in parsed.values()):
                raise ValueError("Judge must return one binary score for every requested item")
        elif any(not isinstance(value, str) for value in parsed.values()):
            raise ValueError("Judge alignment must map strings to strings")
        return response

    def judge_column(response, target, criterion, model_config_name):
        grade = partial(completion, expected_keys={f"idx_{i}" for i in range(len(response))})
        with patch.object(metric_utils, "llm_completion", grade):
            return metric_utils.llm_judge_column(response, target, criterion, model_config_name)

    with patch.object(metric_utils, "llm_completion", completion), patch(
        "src.evaluation.evaluation.llm_judge_column", judge_column
    ):
        result = evaluate_single_query(deepcopy(query), WideSearchResponse(query.instance_id, answer),
                                       str(directory / "cells.csv"), JUDGE_MODEL)
    value = asdict(result)
    value["status"] = "evaluation_error" if result.msg.startswith("evaluator error:") else "evaluated"
    if value["status"] == "evaluation_error":
        for key in ("score", "f1_by_row", "f1_by_item"):
            value[key] = None
    return value


def summarize(manifest, results):
    groups = defaultdict(list)
    for entry in manifest["episodes"]:
        key = f"{entry['task']}-r{entry['repetition']:03d}"
        value = results.get(key, {})
        groups[entry["task"]].append(value if entry["status"] == "finished" and not entry.get("infrastructure_errors") else {})
    n = manifest["repetitions"]
    total = len(manifest["tasks"]) * n
    metrics = {}
    for field in ("score", "f1_by_row", "f1_by_item"):
        values = [[float(row.get(field) or 0) for row in groups[task]] for task in manifest["tasks"]]
        metrics[field] = {"avg": sum(map(sum, values)) / total,
                          "max": sum(max(rows, default=0) for rows in values) / len(values)}
    return {"tasks": len(manifest["tasks"]), "repetitions": n, "scheduled_attempts": total,
            "ended_attempts": len(manifest["episodes"]), "scored_attempts": len(results),
            "status": "completed" if manifest.get("status") == "completed" and all(
                entry["status"] != "finished" or results.get(f"{entry['task']}-r{entry['repetition']:03d}", {}).get("status") == "evaluated"
                for entry in manifest["episodes"]
            ) else "needs_attention",
            "infrastructure_errors": sum(bool(e.get("infrastructure_errors")) for e in manifest["episodes"]),
            "evaluation_errors": sum(v["status"] == "evaluation_error" for v in results.values()),
            "denominator": "all scheduled attempts; missing and failed episodes count as zero",
            "metrics": metrics}


def main(args):
    root = args.run.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    queries = load_queries(args.source)
    directory = root / "evaluation"
    directory.mkdir(exist_ok=True)
    provenance = {"dataset_revision": REVISION,
         "gold_sha256": {task: hashlib.sha256((args.source / "widesearch_gold" / f"{task}.csv").read_bytes()).hexdigest()
                         for task in sorted(queries)},
         "source_sha256": hashlib.sha256((args.source / "widesearch.jsonl").read_bytes()).hexdigest(),
         "upstream_patch_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=UPSTREAM)).hexdigest(),
         "upstream_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip(),
         "judge_model": JUDGE_MODEL,
         "backend": manifest["config"]["retrieval_url"], "harness": "WideSearch"}
    provenance_path = directory / "provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError("Evaluation provenance changed; use a separate evaluation directory")
    save(provenance_path, provenance)
    results = {}
    for entry in manifest["episodes"]:
        if entry.get("infrastructure_errors") or entry["status"] != "finished":
            continue
        key = f"{entry['task']}-r{entry['repetition']:03d}"
        result_path = directory / key / "result.json"
        if result_path.exists():
            results[key] = json.loads(result_path.read_text())
    consecutive_errors = 0
    for entry in manifest["episodes"]:
        if entry.get("infrastructure_errors") or entry["status"] != "finished":
            continue
        key = f"{entry['task']}-r{entry['repetition']:03d}"
        destination = directory / key
        destination.mkdir(exist_ok=True)
        result_path = destination / "result.json"
        if result_path.exists():
            value = json.loads(result_path.read_text())
            if value["status"] == "evaluated":
                results[key] = value
                continue
            archive = directory / "retries"
            archive.mkdir(exist_ok=True)
            destination.rename(archive / f"{key}-{uuid4().hex[:8]}")
            destination.mkdir()
        answer_path = root / entry["directory"] / "answer.md"
        if not answer_path.exists():
            continue
        try:
            value = score(queries[entry["task"]], answer_path.read_text(), destination)
        except Exception as exc:
            value = {"status": "evaluation_error", "score": None, "f1_by_row": None,
                     "f1_by_item": None, "error": repr(exc)}
        save(result_path, value)
        results[key] = value
        save(directory / "summary.json", summarize(manifest, results))
        print(key, value["status"], value["score"], flush=True)
        consecutive_errors = consecutive_errors + 1 if value["status"] == "evaluation_error" else 0
        if consecutive_errors >= 3 or classify(value.get("error") or value.get("msg")) == "credentials_or_quota":
            break
    summary = summarize(manifest, results)
    save(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE)
    main(parser.parse_args())
