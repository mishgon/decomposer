from symtrace.episode_runner import EpisodeRunner


def test_episode_runner_captures_simplify_trace():
    runner = EpisodeRunner(timeout_s=5.0)
    episode = runner.run_episode(
        episode_id="ep_simple",
        entrypoint_name="simplify",
        kwargs={"expr": "(x + x**2)/(x*sin(y)**2 + x*cos(y)**2)"},
        input_seed=1,
    )
    assert episode["episode"]["status"] in {"ok", "trace_truncated"}
    assert episode["episode"]["num_calls"] > 0
    assert episode["episode"]["final_output"]["display"] == "x + 1"
