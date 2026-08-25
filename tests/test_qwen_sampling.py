from gyms.qwen_sampling import qwen35_general_sampling


def test_qwen35_general_sampling_is_mode_specific() -> None:
    non_thinking = qwen35_general_sampling(thinking=False)
    thinking = qwen35_general_sampling(thinking=True)

    assert (
        non_thinking.temperature,
        non_thinking.top_p,
        non_thinking.top_k,
        non_thinking.min_p,
        non_thinking.presence_penalty,
        non_thinking.repetition_penalty,
    ) == (0.7, 0.8, 20, 0.0, 1.5, 1.0)
    assert (
        thinking.temperature,
        thinking.top_p,
        thinking.top_k,
        thinking.min_p,
        thinking.presence_penalty,
        thinking.repetition_penalty,
    ) == (1.0, 0.95, 20, 0.0, 1.5, 1.0)
    assert non_thinking.extra_body == {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
