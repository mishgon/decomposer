"""Generation settings shared by subagents and trace metadata."""

import os

import httpx


def model_http_client() -> httpx.AsyncClient:
    """Optional private socket route; the URL still controls TLS verification."""
    socket = os.environ.get("LLM_PROXY_UNIX_SOCKET")
    limits = httpx.Limits(max_keepalive_connections=0)
    return httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=socket, limits=limits) if socket else None,
        limits=limits,
        trust_env=not bool(socket),
    )


def generation_config(model: str, *, thinking: bool = False) -> dict:
    if "qwen3.5" in model.lower() and not thinking:
        return {
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "preserve_reasoning": False,
            "extra_body": {
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "include_reasoning": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
    extra_body = {"top_k": 64}
    if not thinking:
        extra_body.update(
            reasoning_effort="none",
            chat_template_kwargs={"enable_thinking": False},
        )
    return {
        "temperature": 1.0,
        "top_p": 0.95,
        "preserve_reasoning": thinking,
        "extra_body": extra_body,
    }


def teacher_generation_config(model: str) -> dict:
    settings = {
        "temperature": 1.0,
        "top_p": 0.95,
        "preserve_reasoning": True,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "include_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }
    if "qwen3.8-flash-next" in model.lower():
        settings.update(reasoning_effort="low", presence_penalty=0.0)
        # lmrouter's LiteLLM gateway otherwise rejects this OpenAI parameter for Qwen.
        settings["extra_body"]["allowed_openai_params"] = ["reasoning_effort"]
    return settings
