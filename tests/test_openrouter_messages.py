from eval import DECOMPOSER_MODEL, QA_MODELS
from prototype import DEFAULT_DECOMPOSER_MODEL, DEFAULT_QA_MODEL


def test_qwen36_defaults_are_used_in_active_entrypoints():
    assert DEFAULT_DECOMPOSER_MODEL == "qwen/qwen3.6-27b"
    assert DEFAULT_QA_MODEL == "qwen/qwen3.6-27b"
    assert DECOMPOSER_MODEL == "Qwen/Qwen3.6-27B"
    assert QA_MODELS == ("Qwen/Qwen3.6-27B",)
