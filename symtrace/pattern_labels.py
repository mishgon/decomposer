from __future__ import annotations

from collections import Counter

from .schemas import CallNode


def label_trace_pattern(nodes: dict[str, CallNode], root_call_id: str | None) -> str:
    if not root_call_id or not nodes:
        return "chain"
    branching = [len(node.subcalls) for node in nodes.values()]
    repeated = Counter(node.func_id for node in nodes.values()).most_common(1)[0][1]
    max_branch = max(branching, default=0)
    if max_branch >= 2 and repeated >= 3:
        return "mixed"
    if max_branch <= 1 and repeated <= 2:
        return "chain"
    if repeated >= 3 and max_branch <= 2:
        return "loop"
    if max_branch >= 2:
        return "hierarchy"
    return "mixed"
