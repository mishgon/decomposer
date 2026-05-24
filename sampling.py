QWEN3_THINKING_SAMPLING_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens": 8192,
}


QWEN3_NON_THINKING_SAMPLING_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_tokens": 8192,
}


QWEN36_QA_THINKING_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens": 8192,
}


QWEN36_CODE_THINKING_SAMPLING_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens": 8192,
}


QWEN36_NON_THINKING_SAMPLING_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_tokens": 8192,
}


def build_sampling_params(
    model: str,
    enable_thinking: bool,
    task: str,
) -> dict:
    if task not in {"code", "general"}:
        raise ValueError(f"Unknown sampling task {task!r}.")

    model_name = model.lower()
    if "qwen3.6-" in model_name:
        if not enable_thinking:
            return QWEN36_NON_THINKING_SAMPLING_PARAMS.copy()
        if task == "code":
            return QWEN36_CODE_THINKING_SAMPLING_PARAMS.copy()
        return QWEN36_QA_THINKING_SAMPLING_PARAMS.copy()
    if "qwen3-" in model_name:
        if enable_thinking:
            return QWEN3_THINKING_SAMPLING_PARAMS.copy()
        return QWEN3_NON_THINKING_SAMPLING_PARAMS.copy()
    raise ValueError(f"No sampling parameters configured for model {model!r}.")
