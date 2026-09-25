from gyms.qwen_sampling import (
    qwen35_general_sampling,
    qwen36_non_thinking_sampling,
    qwen36_thinking_sampling,
)


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


def test_qwen36_non_thinking_sampling_matches_instruct_recommendation() -> None:
    sampling = qwen36_non_thinking_sampling()

    assert (
        sampling.temperature,
        sampling.top_p,
        sampling.top_k,
        sampling.min_p,
        sampling.presence_penalty,
        sampling.repetition_penalty,
    ) == (0.7, 0.8, 20, 0.0, 1.5, 1.0)


def test_qwen36_thinking_sampling_matches_general_recommendation() -> None:
    sampling = qwen36_thinking_sampling()

    assert (
        sampling.temperature,
        sampling.top_p,
        sampling.top_k,
        sampling.min_p,
        sampling.presence_penalty,
        sampling.repetition_penalty,
    ) == (1.0, 0.95, 20, 0.0, 1.5, 1.0)


def test_subagent_sampling_defaults_to_the_qwen35_non_thinking_preset() -> None:
    from gyms.qwen_sampling import (
        non_thinking_subagent_sampling_kwargs,
        subagent_sampling_environment,
    )

    assert subagent_sampling_environment(None) == {}
    # Exactly what the subagent graphs sent before the setting existed.
    assert non_thinking_subagent_sampling_kwargs({}) == {
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def test_unlooped_subagent_sampling_sends_only_what_is_set() -> None:
    import pytest

    from gyms.qwen_sampling import (
        QWEN35_UNLOOPED_NON_THINKING,
        SUBAGENT_SAMPLING_ENV,
        non_thinking_subagent_sampling_kwargs,
        subagent_sampling_environment,
    )

    environment = subagent_sampling_environment(QWEN35_UNLOOPED_NON_THINKING)
    assert environment["DECOMPOSER_SUBAGENT_MAX_COMPLETION_TOKENS"] == "2048"
    # No penalties and no min_p: the unlooped model was evaluated without them.
    assert non_thinking_subagent_sampling_kwargs(environment) == {
        "temperature": 0.7,
        "top_p": 0.8,
        "extra_body": {
            "top_k": 20,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    with pytest.raises(ValueError, match="unknown keys"):
        non_thinking_subagent_sampling_kwargs(
            {SUBAGENT_SAMPLING_ENV: '{"temperature": 1, "top_p": 1, "top_k": 1, "typo": 1}'}
        )


def test_qwen38_teacher_thinking_has_no_presence_penalty() -> None:
    from gyms.qwen_sampling import QWEN38_TEACHER_THINKING, QWEN38_THINKING

    assert (
        QWEN38_TEACHER_THINKING.temperature,
        QWEN38_TEACHER_THINKING.top_p,
        QWEN38_TEACHER_THINKING.top_k,
        QWEN38_TEACHER_THINKING.min_p,
        QWEN38_TEACHER_THINKING.presence_penalty,
        QWEN38_TEACHER_THINKING.repetition_penalty,
    ) == (1.0, 0.95, 20, 0.0, 0.0, 1.0)
    assert QWEN38_THINKING.presence_penalty == 1.5
