import json

from symtrace.episode_runner import EpisodeRunner
from symtrace.sft.export_traces import export_training_example


def test_export_training_example(tmp_path):
    runner = EpisodeRunner(timeout_s=5.0)
    raw_episode = runner.run_episode(
        episode_id="ep_export",
        entrypoint_name="solve",
        kwargs={"expr": "x**2 - 1", "symbol_names": ["x"]},
        input_seed=1,
    )
    output = tmp_path / "train.jsonl"
    export_training_example(raw_episode, output)
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["task"]["entry_function"] == "solve"
