"""The gyms' bridge to the model presets in ``src/decomposer/models.py``.

``create_model(preset_id)`` is the single source of a model's sampling. Subagents use
it as is: their LangGraph graphs call ``create_model`` and reach the shared proxy on
the preset's own endpoint. The Decomposer manager cannot: it runs inside NeMo Gym's
``openai_model`` server on the Responses API, behind ``gyms.remote_model_proxy``. For it,
``manager_responses_body`` turns the preset into the extra body that proxy merges into
every request. On the Responses API the shared proxy ignores ``chat_template_kwargs``,
so thinking is switched by ``reasoning.effort`` there: "none" for a non-thinking preset.

Older experiments keep their values in ``gyms/qwen_sampling.py``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from decomposer.models import ChatVLLM, create_model

QWEN38_FLASH_NON_THINKING_PRESET = "lmrouter/qwen_3_8_flash_next_non_thinking"
QWEN35_UNLOOPED_THINKING_PRESET = "lmrouter/qwen_3_5_4b_unlooped_thinking"

# Subagent presets and the LangGraph graph serving each, in both gyms' langgraph.json.
SUBAGENT_PRESET_GRAPHS = {QWEN35_UNLOOPED_THINKING_PRESET: "qwen35_4b_unlooped_thinking"}

# Chat Completions extra_body keys a preset may carry, and what the manager body keeps.
_SAMPLING_KEYS = ("top_k", "min_p", "repetition_penalty")
_KNOWN_EXTRA_BODY_KEYS = {*_SAMPLING_KEYS, "chat_template_kwargs", "allowed_openai_params"}


@lru_cache(maxsize=None)
def _preset(preset_id: str) -> ChatVLLM:
    if not preset_id.startswith("lmrouter/"):
        raise ValueError(f"{preset_id}: only lmrouter presets are supported")
    model = create_model(preset_id)  # type: ignore[arg-type]
    if not isinstance(model, ChatVLLM):
        raise TypeError(f"{preset_id}: expected a ChatVLLM, got {type(model).__name__}")
    unknown = sorted(set(model.extra_body or {}) - _KNOWN_EXTRA_BODY_KEYS)
    if unknown:
        raise ValueError(f"{preset_id}: extra_body keys {unknown} have no Responses API mapping")
    return model


def upstream_model_id(preset_id: str) -> str:
    """The model the preset requests from the shared proxy."""
    return _preset(preset_id).model_name


def is_thinking(preset_id: str) -> bool:
    template = (_preset(preset_id).extra_body or {}).get("chat_template_kwargs") or {}
    return bool(template.get("enable_thinking"))


def manager_responses_body(preset_id: str) -> dict[str, Any]:
    """The preset as ``gyms.remote_model_proxy --extra-body-json`` for a Responses API manager."""
    model = _preset(preset_id)
    extra = model.extra_body or {}
    thinking = is_thinking(preset_id)
    body: dict[str, Any] = {"temperature": model.temperature, "top_p": model.top_p}
    if model.presence_penalty is not None:
        body["presence_penalty"] = model.presence_penalty
    body.update({key: extra[key] for key in _SAMPLING_KEYS if key in extra})
    body["include_reasoning"] = thinking
    body["chat_template_kwargs"] = dict(extra.get("chat_template_kwargs") or {})
    if not thinking:
        body["reasoning"] = {"effort": "none"}
    elif model.reasoning_effort is not None:
        body["reasoning"] = {"effort": model.reasoning_effort}
    return body


def preset_record(preset_id: str) -> dict[str, Any]:
    """What run_status records about a preset: its parameters, never its endpoint or key."""
    model = _preset(preset_id)
    return {
        "preset": preset_id,
        "model": model.model_name,
        "temperature": model.temperature,
        "top_p": model.top_p,
        "presence_penalty": model.presence_penalty,
        "reasoning_effort": model.reasoning_effort,
        "extra_body": dict(model.extra_body or {}),
        "preserve_reasoning": model.preserve_reasoning,
        "max_completion_tokens": model.max_tokens,
        "timeout_seconds": model.request_timeout,
        "max_retries": model.max_retries,
    }
