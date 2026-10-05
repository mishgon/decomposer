"""Public Decomposer API with lazy imports for lightweight CLI consumers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_decomposer_chat_tools",
    "create_decomposer_agent",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    if name == "build_decomposer_chat_tools":
        from .chat_tools import build_decomposer_chat_tools

        return build_decomposer_chat_tools
    from .core import create_decomposer_agent

    return create_decomposer_agent
