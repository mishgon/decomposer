"""Interpret native evaluator outputs without assigning sampling policy."""

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PartialScore:
    passed_checks: int
    total_checks: int
    source: str

    @property
    def fraction(self) -> float:
        return self.passed_checks / self.total_checks


def _numeric_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or int(value) != value:
        return None
    return int(value)


def extract_partial_score(evaluation: dict[str, Any]) -> PartialScore | None:
    """Extract native check counts without guessing from arbitrary log lines."""
    native = evaluation.get("native_result")
    if isinstance(native, dict):
        passed = _numeric_count(native.get("total_passed"))
        total = _numeric_count(native.get("total_checks"))
        if passed is not None and total and passed <= total:
            return PartialScore(passed, total, "native_total")

        passed = _numeric_count(native.get("passed"))
        total = _numeric_count(native.get("total"))
        if passed is not None and total and passed <= total:
            return PartialScore(passed, total, "native_total")

        for passed_key, failed_key in (("passed", "failed"), ("pass", "fail")):
            passed = _numeric_count(native.get(passed_key))
            failed = _numeric_count(native.get(failed_key))
            if passed is not None and failed is not None and passed + failed > 0:
                return PartialScore(passed, passed + failed, "native_pass_fail")

    stdout = evaluation.get("stdout")
    if not isinstance(stdout, str):
        return None

    fraction_patterns = (
        r"(?:Results:\s*)?(\d+)\s*/\s*(\d+)\s+passed",
        r"Passed\s+(\d+)\s*/\s*(\d+)\s+checks",
    )
    for pattern in fraction_patterns:
        matches = re.findall(pattern, stdout, flags=re.IGNORECASE)
        if matches:
            passed, total = map(int, matches[-1])
            if total > 0 and passed <= total:
                return PartialScore(passed, total, "stdout_fraction")

    pass_fail_patterns = (
        r"Passed\s*:?\s*(\d+)\s*(?:,|\n)\s*Failed\s*:?\s*(\d+)",
        r"(\d+)\s+passed\s*,\s*(\d+)\s+failed",
    )
    for pattern in pass_fail_patterns:
        matches = re.findall(pattern, stdout, flags=re.IGNORECASE)
        if matches:
            passed, failed = map(int, matches[-1])
            if passed + failed > 0:
                return PartialScore(passed, passed + failed, "stdout_pass_fail")

    # A failed evaluator may have crashed after its first passing check. Without
    # an explicit total above, its log cannot establish the score denominator.
    if evaluation.get("returncode") != 0:
        return None

    # Some native evaluators only emit one line per check. Keep this fallback
    # deliberately narrow: bracketed check markers and standalone status lines
    # are unambiguous, while arbitrary occurrences of words like "error" are not.
    passed = len(re.findall(r"^\s*\[(?:PASS|OK)\]", stdout, re.MULTILINE))
    failed = len(re.findall(r"^\s*\[(?:FAIL|ERROR)\]", stdout, re.MULTILINE))
    if passed + failed > 0:
        return PartialScore(passed, passed + failed, "stdout_check_markers")

    passed = len(re.findall(r"^\s*PASS\s*$", stdout, re.MULTILINE))
    failed = len(re.findall(r"^\s*FAIL\s*$", stdout, re.MULTILINE))
    if passed + failed > 0:
        return PartialScore(passed, passed + failed, "stdout_status_lines")
    return None
