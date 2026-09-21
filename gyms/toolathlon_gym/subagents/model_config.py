"""Generation settings shared by subagents and trace metadata."""


def generation_config(model: str, *, thinking: bool = False) -> dict:
    if any(family in model.lower() for family in ("qwen3.5", "qwen3.8")) and not thinking:
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
