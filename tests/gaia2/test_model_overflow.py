from __future__ import annotations

import asyncio

import pytest
from langchain.agents.middleware.model_call_limit import (
    ModelCallLimitExceededError,
)
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from gyms.gaia2.model_overflow import (
    ExactModelCallLimitMiddleware,
    Gaia2ModelOverflowError,
    Gaia2ModelOverflowMiddleware,
    is_context_overflow_error,
    output_overflow_detail,
)


def test_exact_call_limiter_reserves_before_dispatch() -> None:
    limiter = ExactModelCallLimitMiddleware(run_limit=2, exit_behavior="error")

    first = limiter.before_model({}, None)
    second = limiter.before_model(first, None)

    assert first == {"thread_model_call_count": 1, "run_model_call_count": 1}
    assert second == {"thread_model_call_count": 2, "run_model_call_count": 2}
    assert limiter.after_model(second, None) is None
    with pytest.raises(ModelCallLimitExceededError):
        limiter.before_model(second, None)


def test_context_overflow_recognizes_provider_class_and_message() -> None:
    ContextWindowExceededError = type("ContextWindowExceededError", (RuntimeError,), {})

    assert is_context_overflow_error(ContextWindowExceededError("rejected"))
    assert is_context_overflow_error(
        RuntimeError("This model's maximum context length is 131072 tokens")
    )
    assert not is_context_overflow_error(RuntimeError("unrelated HTTP 400"))


@pytest.mark.parametrize(
    "metadata",
    [
        {"finish_reason": "length"},
        {"stop_reason": "max_tokens"},
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        },
    ],
)
def test_output_overflow_recognizes_chat_and_responses_metadata(metadata) -> None:
    response = ModelResponse(
        result=[AIMessage(content="truncated", response_metadata=metadata)]
    )

    assert output_overflow_detail(response) is not None


def test_sync_middleware_fails_context_overflow_without_retry() -> None:
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        raise RuntimeError("maximum context length is 131072 tokens")

    middleware = Gaia2ModelOverflowMiddleware(
        "manager", max_completion_tokens=8192, max_model_len=131072
    )
    with pytest.raises(Gaia2ModelOverflowError) as captured:
        middleware.wrap_model_call(object(), handler)

    assert calls == 1
    assert captured.value.as_dict() == {
        "actor": "manager",
        "kind": "input_context_overflow",
        "detail": "maximum context length is 131072 tokens",
        "policy": "fail_actor_v1",
        "max_completion_tokens": 8192,
        "max_model_len": 131072,
    }


def test_async_middleware_discards_truncated_response_without_retry() -> None:
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        return ModelResponse(
            result=[
                AIMessage(
                    content="partial",
                    tool_calls=[
                        {"name": "dangerous", "args": {}, "id": "call-1"}
                    ],
                    response_metadata={"finish_reason": "length"},
                )
            ]
        )

    middleware = Gaia2ModelOverflowMiddleware("subagent")
    with pytest.raises(Gaia2ModelOverflowError) as captured:
        asyncio.run(middleware.awrap_model_call(object(), handler))

    assert calls == 1
    assert captured.value.kind == "output_token_overflow"
    assert captured.value.actor == "subagent"


def test_overflow_detail_is_bounded_in_artifacts() -> None:
    error = Gaia2ModelOverflowError(
        actor="manager",
        kind="input_context_overflow",
        detail="x" * 5000,
    )

    assert len(error.detail) == 800
    assert error.detail.endswith("...")
