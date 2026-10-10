"""Fail-fast classification for GAIA2 policy-model overflows."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage

OverflowKind = Literal["input_context_overflow", "output_token_overflow"]

_CONTEXT_ERROR_MARKERS = (
    "context length exceeded",
    "context window exceeded",
    "exceeded its context window",
    "maximum context length",
    "prompt is too long",
    "prompt too long",
)
_OUTPUT_FINISH_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})


def _bounded_message(value: Any, limit: int = 800) -> str:
    message = " ".join(str(value).split())
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


class Gaia2ModelOverflowError(RuntimeError):
    """A non-retryable input- or output-token overflow for one GAIA2 actor."""

    def __init__(
        self,
        *,
        actor: str,
        kind: OverflowKind,
        detail: Any,
        max_completion_tokens: int | None = None,
        max_model_len: int | None = None,
    ) -> None:
        self.actor = actor
        self.kind = kind
        self.detail = _bounded_message(detail)
        self.max_completion_tokens = max_completion_tokens
        self.max_model_len = max_model_len
        super().__init__(f"GAIA2 {actor} {kind}: {self.detail}")

    def as_dict(self) -> dict[str, str | int]:
        value: dict[str, str | int] = {
            "actor": self.actor,
            "kind": self.kind,
            "detail": self.detail,
            "policy": "fail_actor_v1",
        }
        if self.max_completion_tokens is not None:
            value["max_completion_tokens"] = self.max_completion_tokens
        if self.max_model_len is not None:
            value["max_model_len"] = self.max_model_len
        return value


def _exception_chain(error: BaseException):
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        nested = getattr(current, "inner_exception", None)
        if not isinstance(nested, BaseException):
            nested = current.__cause__ or current.__context__
        current = nested if isinstance(nested, BaseException) else None


def is_context_overflow_error(error: BaseException) -> bool:
    """Recognize OpenAI, LiteLLM, vLLM, and proxy context-limit failures."""

    for current in _exception_chain(error):
        if type(current).__name__ == "ContextWindowExceededError":
            return True
        body = getattr(current, "body", None)
        if isinstance(body, Mapping):
            nested = body.get("error")
            if isinstance(nested, Mapping):
                body = nested
            code = str(body.get("code", "")).lower()
            if code in {"context_length_exceeded", "context_window_exceeded"}:
                return True
        message = str(current).lower()
        if any(marker in message for marker in _CONTEXT_ERROR_MARKERS):
            return True
    return False


def _response_messages(response: Any) -> list[AIMessage]:
    if isinstance(response, AIMessage):
        return [response]
    result = getattr(response, "result", None)
    if not isinstance(result, list):
        return []
    return [message for message in result if isinstance(message, AIMessage)]


def output_overflow_detail(response: Any) -> str | None:
    """Return the provider's output-limit reason, if the response was truncated."""

    for message in _response_messages(response):
        metadata = message.response_metadata or {}
        finish_reason = metadata.get("finish_reason") or metadata.get("stop_reason")
        if str(finish_reason).lower() in _OUTPUT_FINISH_REASONS:
            return f"finish_reason={finish_reason}"

        incomplete = metadata.get("incomplete_details")
        reason = incomplete.get("reason") if isinstance(incomplete, Mapping) else None
        if str(reason).lower() in _OUTPUT_FINISH_REASONS:
            return f"incomplete_details.reason={reason}"

    return None


class Gaia2ModelOverflowMiddleware(AgentMiddleware):
    """Abort one actor immediately when its model input or output overflows."""

    def __init__(
        self,
        actor: str,
        *,
        max_completion_tokens: int | None = None,
        max_model_len: int | None = None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.max_completion_tokens = max_completion_tokens
        self.max_model_len = max_model_len

    def _error(self, *, kind: OverflowKind, detail: Any) -> Gaia2ModelOverflowError:
        return Gaia2ModelOverflowError(
            actor=self.actor,
            kind=kind,
            detail=detail,
            max_completion_tokens=self.max_completion_tokens,
            max_model_len=self.max_model_len,
        )

    def _check_response(self, response: Any) -> Any:
        detail = output_overflow_detail(response)
        if detail is not None:
            raise self._error(
                kind="output_token_overflow",
                detail=detail,
            )
        return response

    def _translate_error(self, error: BaseException) -> None:
        if is_context_overflow_error(error):
            raise self._error(
                kind="input_context_overflow",
                detail=error,
            ) from error

    def wrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        try:
            response = handler(request)
        except Exception as error:
            self._translate_error(error)
            raise
        return self._check_response(response)

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        try:
            response = await handler(request)
        except Exception as error:
            self._translate_error(error)
            raise
        return self._check_response(response)


class ExactModelCallLimitMiddleware(ModelCallLimitMiddleware):
    """Reserve a model call before dispatch so failed requests count too."""

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        terminal = super().before_model(state, runtime)
        if terminal is not None:
            return terminal
        return {
            "thread_model_call_count": state.get("thread_model_call_count", 0) + 1,
            "run_model_call_count": state.get("run_model_call_count", 0) + 1,
        }

    def after_model(self, state: Any, runtime: Any) -> None:
        return None
