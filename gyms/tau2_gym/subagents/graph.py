"""LangGraph subagents for the tau2 gym.

Subagents are compiled with ``tools=[]``: the tau2 tool schemas ride in on the
dataset row and are injected per run by ``NeMoGymSubagentMiddleware``, which also
routes each call to ``POST {resource_server_url}/{tool_name}``
(``decomposer_agent/subagents/graph.py:37-42,83-90``). So nothing tau2-specific
belongs here -- this module only pins the model endpoints.

Only Qwen3.5-4B is registered: manager and subagents share one architecture, but
they must be served from *separate* vLLM instances, with the subagent pinned to a
frozen checkpoint. Serving both from one process would make the environment
non-stationary the moment the manager is trained.
"""

from __future__ import annotations

import json
import os

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from responses_api_agents.decomposer_agent.subagents.graph import (
    NeMoGymSubagentMiddleware,
)

from decomposer.models import ChatVLLM, create_model
from decomposer.prompts import AGENT_SYSTEM_PROMPT
from gyms.model_presets import QWEN35_UNLOOPED_THINKING_PRESET
from gyms.qwen_sampling import non_thinking_subagent_sampling_kwargs
from gyms.tau2_gym.experiments import DEFAULT_SUBAGENT_MODEL_ID, SUBAGENT_MODEL_ENV

REQUEST_TIMEOUT_SECONDS = 300.0
MODEL_BASE_URLS_ENV = "TAU2_GYM_MODEL_BASE_URLS_JSON"
# The model the subagents request; run.py sets it per run (`--subagent-model-id`).
QWEN35_4B_MODEL_ID = os.environ.get(SUBAGENT_MODEL_ENV) or DEFAULT_SUBAGENT_MODEL_ID
QWEN35_4B_DEFAULT_PORT = 8025

SUBAGENT_MAX_MODEL_CALLS = int(os.environ.get("DECOMPOSER_SUBAGENT_MAX_MODEL_CALLS", "100"))
if SUBAGENT_MAX_MODEL_CALLS < 1:
    raise ValueError("DECOMPOSER_SUBAGENT_MAX_MODEL_CALLS must be at least 1")

_max_completion_tokens = os.environ.get("DECOMPOSER_SUBAGENT_MAX_COMPLETION_TOKENS")
SUBAGENT_MAX_COMPLETION_TOKENS = (
    int(_max_completion_tokens) if _max_completion_tokens is not None else None
)
if SUBAGENT_MAX_COMPLETION_TOKENS is not None and SUBAGENT_MAX_COMPLETION_TOKENS < 1:
    raise ValueError("DECOMPOSER_SUBAGENT_MAX_COMPLETION_TOKENS must be at least 1")


def _completion_kwargs() -> dict[str, int]:
    if SUBAGENT_MAX_COMPLETION_TOKENS is None:
        return {}
    return {"max_completion_tokens": SUBAGENT_MAX_COMPLETION_TOKENS}


def _model_base_url(model_id: str, default_port: int) -> str:
    raw = os.environ.get(MODEL_BASE_URLS_ENV)
    if raw is None:
        return f"http://127.0.0.1:{default_port}/v1"
    try:
        configured = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{MODEL_BASE_URLS_ENV} must contain valid JSON") from error
    if not isinstance(configured, dict):
        raise ValueError(f"{MODEL_BASE_URLS_ENV} must contain a JSON object")
    value = configured.get(model_id)
    if value is None:
        return f"http://127.0.0.1:{default_port}/v1"
    if not isinstance(value, str) or not value:
        raise ValueError(f"{MODEL_BASE_URLS_ENV}[{model_id!r}] must be a URL string")
    return value


def _create_subagent(model: BaseChatModel) -> CompiledStateGraph:
    return create_agent(
        model=model,
        tools=[],
        middleware=[
            NeMoGymSubagentMiddleware(),
            ModelCallLimitMiddleware(
                run_limit=SUBAGENT_MAX_MODEL_CALLS,
                exit_behavior="error",
            ),
        ],
        system_prompt=AGENT_SYSTEM_PROMPT,
    )


def qwen35_4b_non_thinking() -> CompiledStateGraph:
    # The experiment's sampling (DECOMPOSER_SUBAGENT_SAMPLING_JSON, set by run.py),
    # else Qwen3.5's general non-thinking preset.
    model = ChatVLLM(
        model=QWEN35_4B_MODEL_ID,
        base_url=_model_base_url(QWEN35_4B_MODEL_ID, QWEN35_4B_DEFAULT_PORT),
        api_key="EMPTY",
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=False,
        **_completion_kwargs(),
        **non_thinking_subagent_sampling_kwargs(),
    )
    return _create_subagent(model)


def qwen35_4b_unlooped_thinking() -> CompiledStateGraph:
    # The models.py preset as is: its sampling, thinking with reasoning kept across
    # turns, no output cap, and its own proxy endpoint (LLM_PROXY_MASTER_KEY).
    return _create_subagent(create_model(QWEN35_UNLOOPED_THINKING_PRESET))
