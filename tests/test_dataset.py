import json

from symtrace.dataset import export_demo_corpus


def test_export_demo_corpus(tmp_path):
    summary = export_demo_corpus(tmp_path, num_expr=1, num_solve=1)
    assert summary["episodes"] == 4
    assert summary["catalog_entries"] == 4
    function_catalog = (tmp_path / "function_catalog.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(function_catalog) == 4
    raw_files = list((tmp_path / "raw_traces").rglob("*.json"))
    assert len(raw_files) == 4
    abstract_path = tmp_path / "abstract_traces" / "train.jsonl"
    assert abstract_path.exists()
    records = [json.loads(line) for line in abstract_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records
    assert (tmp_path / "debug_samples_guide.md").exists()
    assert (tmp_path / "chain_examples.md").exists()
    assert (tmp_path / "ideal_reasoning_examples.md").exists()
