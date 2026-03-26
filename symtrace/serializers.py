from __future__ import annotations

import base64
import pickle
from collections.abc import Mapping, Sequence
import inspect

from sympy import Basic
from sympy.printing import srepr


def _safe_pickle(value: object) -> str | None:
    try:
        raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        return None
    return base64.b64encode(raw).decode("ascii")


def _safe_text(fn, fallback: str) -> str:
    try:
        value = fn()
    except Exception:
        return fallback
    return value


def serialize_value(value: object, *, max_string: int = 500) -> dict:
    if isinstance(value, Basic):
        fallback = f"<unprintable {type(value).__module__}.{type(value).__name__}>"
        display = _safe_text(lambda: str(value), fallback)
        rep = _safe_text(lambda: repr(value), fallback)
        sexpr = _safe_text(lambda: srepr(value), fallback)
        return {
            "type_tag": "sympy.Basic",
            "display": display[:max_string],
            "repr": rep[:max_string],
            "srepr": sexpr[:max_string],
            "python_type": f"{type(value).__module__}.{type(value).__name__}",
            "pickle_b64": _safe_pickle(value),
        }

    if isinstance(value, Mapping):
        items = [
            {"key": serialize_value(k, max_string=max_string), "value": serialize_value(v, max_string=max_string)}
            for k, v in list(value.items())[:50]
        ]
        return {
            "type_tag": "mapping",
            "display": _safe_text(lambda: str(value), f"<unprintable {type(value).__name__}>")[:max_string],
            "repr": _safe_text(lambda: repr(value), f"<unprintable {type(value).__name__}>")[:max_string],
            "python_type": f"{type(value).__module__}.{type(value).__name__}",
            "items": items,
            "pickle_b64": _safe_pickle(value),
        }

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [serialize_value(item, max_string=max_string) for item in list(value)[:50]]
        return {
            "type_tag": "sequence",
            "display": _safe_text(lambda: str(value), f"<unprintable {type(value).__name__}>")[:max_string],
            "repr": _safe_text(lambda: repr(value), f"<unprintable {type(value).__name__}>")[:max_string],
            "python_type": f"{type(value).__module__}.{type(value).__name__}",
            "items": items,
            "pickle_b64": _safe_pickle(value),
        }

    return {
        "type_tag": "python",
        "display": _safe_text(lambda: str(value), f"<unprintable {type(value).__name__}>")[:max_string],
        "repr": _safe_text(lambda: repr(value), f"<unprintable {type(value).__name__}>")[:max_string],
        "python_type": f"{type(value).__module__}.{type(value).__name__}",
        "pickle_b64": _safe_pickle(value),
    }


def serialize_frame_locals(local_vars: dict[str, object], *, max_items: int = 20) -> dict[str, dict]:
    serialized: dict[str, dict] = {}
    for name in list(local_vars)[:max_items]:
        serialized[name] = serialize_value(local_vars[name])
    return serialized


def serialize_call_arguments(frame, *, max_items: int = 20) -> dict[str, dict]:
    arg_info = inspect.getargvalues(frame)
    names = list(arg_info.args)
    if arg_info.varargs:
        names.append(arg_info.varargs)
    if arg_info.keywords:
        names.append(arg_info.keywords)
    serialized: dict[str, dict] = {}
    for name in names[:max_items]:
        if name in arg_info.locals:
            serialized[name] = serialize_value(arg_info.locals[name])
    return serialized
