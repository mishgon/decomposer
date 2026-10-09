"""No models, shell, arbitrary paths, or timing-dependent reports."""
import fcntl
import json
from pathlib import Path
import re

COMMAND = re.compile(r"(COPY|CHECK) ([a-z]{8}\.txt) ([a-z]{8}\.txt)")


def parse(prompt):
    lines = [line.strip() for line in prompt.strip().splitlines() if line.strip()]
    matches = [COMMAND.fullmatch(line) for line in lines]
    if not lines or any(match is None for match in matches):
        raise ValueError("INVALID_COMMAND: use COPY source.txt target.txt or CHECK target.txt original_source.txt")
    return [match.groups() for match in matches]


def execute(state, prompt):
    try:
        commands = parse(prompt)
    except ValueError as error:
        return str(error)
    if any(a not in state["files"] or b not in state["files"] for _, a, b in commands):
        return "UNKNOWN_FILE: only the six task files exist"
    reports = []
    for operation, a, b in commands:
        if operation == "COPY":
            state["files"][b] = state["files"][a]
            reports.append(f"COPIED {a} {b}")
        else:
            passed = state["files"][a] == state["initial"][b]
            reports.append(f"CHECK {a} {b} {'PASS' if passed else 'FAIL'}")
    return "\n".join(reports)


def initialize(directory, task):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "workspace.json").write_text(json.dumps({
        "initial": task["initial"], "files": task["initial"].copy(), "events": []}))


def execute_saved(directory, prompt, agent_id, run_id):
    directory = Path(directory).resolve()
    with (directory / "workspace.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / "workspace.json"
        state = json.loads(path.read_text())
        report = execute(state, prompt)
        state["events"].append({"agent_id": agent_id, "run_id": run_id, "prompt": prompt, "report": report})
        temporary = directory / "workspace.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(path)
    return report


def grade(task, state, agent_runs):
    correct = sum(state["files"][name] == expected for name, expected in task["expected"].items())
    copies = 0
    errors = 0
    for event in state["events"]:
        errors += event["report"].startswith(("INVALID_COMMAND", "UNKNOWN_FILE"))
        copies += sum(line.startswith("COPIED ") for line in event["report"].splitlines())
    # Logical concurrency: launched copy runs not yet collected via wait.
    boundaries = []
    for run in agent_runs.values():
        try:
            commands = parse(run["prompt"])
        except ValueError:
            continue
        if any(command[0] == "COPY" for command in commands) and "collected_at" in run:
            boundaries.extend([(run["started_at"], 1), (run["collected_at"], -1)])
    active = peak = 0
    for _, change in sorted(boundaries):
        active += change
        peak = max(peak, active)
    return {"score": correct / 4, "correct_files": correct, "copies": copies,
            "invalid_runs": errors, "peak_unawaited_copy_runs": peak,
            "parallel": peak >= 2, "minimum_copies": 6}
