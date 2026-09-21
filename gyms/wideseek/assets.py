"""Download pinned offline assets; safe to resume with the Hugging Face cache."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

CORPUS_REVISION = "178d7d037f661be3159b0c3a8a4119b974f01880"
RLINF_REVISION = "64875d346d5cafb06c1112f563b20d2c6360bfae"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = args.root / "assets.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["corpus_revision"] != CORPUS_REVISION or saved["rlinf_revision"] != RLINF_REVISION:
            raise ValueError("Existing assets use another revision; choose a new directory")
        print("Offline assets already prepared; not overwriting a potentially live Qdrant store")
        raise SystemExit(0)
    snapshot_download("RLinf/Wiki-2018-Corpus", repo_type="dataset", revision=CORPUS_REVISION,
                      local_dir=args.root / "Wiki-2018-Corpus", max_workers=4)
    # E5's model revision is recorded after the immutable snapshot resolves.
    from huggingface_hub import model_info
    revision_file = args.root / "e5-revision.txt"
    revision = revision_file.read_text().strip() if revision_file.exists() else model_info("intfloat/e5-base-v2").sha
    if not revision_file.exists():
        revision_file.write_text(revision + "\n")
    snapshot_download("intfloat/e5-base-v2", revision=revision,
                      local_dir=args.root / "e5-base-v2",
                      allow_patterns=["*.json", "*.txt", "model.safetensors", "pytorch_model.bin"], max_workers=2)
    manifest.write_text(json.dumps({
        "corpus": "RLinf/Wiki-2018-Corpus", "corpus_revision": CORPUS_REVISION,
        "encoder": "intfloat/e5-base-v2", "encoder_revision": revision,
        "rlinf_revision": RLINF_REVISION}, indent=2) + "\n")
    print("Offline assets ready", flush=True)
