# WideSearch With OCC Researchers

Before running or scoring, prepare the pinned upstream checkout:

```bash
bash gyms/widesearch/setup.sh
```

This applies the tracked evaluator fix for tables with no matching rows.
The setup is safe to repeat and fails if the checkout cannot accept the patch.

This experiment uses WideSearch questions and gold tables with the same OCC-4B
non-thinking researcher as BrowseComp. It reuses the WideSeek agentic-rag prompt,
search/open/find/finish tools, and 32-call limit. Workers have no time budget;
the episode timeout still applies. Single agents and workers use the same harness.

Load `LLM_PROXY_MASTER_KEY` and `AGENTIC_RAG_API_KEY`. The default Search API is
`https://lmrouter.2a2i.org/rag` (`source=wikipedia`), overridable with
`AGENTIC_RAG_URL` or `--retrieval-url`. This is a corpus-backed experiment,
not the upstream live-web benchmark. Keenable is not used.

The shared research protocol counts context tokens with the public
`Qwen/Qwen3.5-4B` tokenizer (no model weights are loaded). Install this gym's
requirements and cache the tokenizer before running offline:

```bash
uv pip install -r gyms/widesearch/requirements.txt
python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")'
```

Use `--agent researcher_occ_non_thinking` for the OCC single-agent baseline.

```bash
PYTHONPATH=src:. python -m gyms.widesearch.run \
  --agent decomposer_sft_1_4_0 --tasks ws_en_001 \
  --concurrency 1 -n 1
PYTHONPATH=src:. python -m evals.widesearch.run PATH_TO_RUN
```

Answers are saved verbatim in `answer.md`. The task requests a table in a fenced
`markdown` block, using the requested column names. Offline evaluation retains
the pinned upstream strict success, row F1 and item F1 metrics and records judge
failures explicitly. Model credentials are loaded by the launcher, not the CLI.
Use a fresh output directory when changing the harness or retrieval backend.
