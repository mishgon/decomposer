# Stock Qwen3.5-4B Baseline

With the existing OPD/vLLM environment active on Hertz-2:

```bash
bash evals/synth/baseline.sh 0 "$HOME/models/Qwen3.5-4B" \
  "$PWD/artifacts/evals/synth/stock-qwen4b"
```

This uses one otherwise idle GPU, prefix caching, a 32K server context, and the
registry's stock Qwen3.5-4B non-thinking settings. The model is **not unlooped or
Decomposer SFT**. The script stops its own vLLM server when evaluation exits.
It refuses an occupied GPU or inference port. It runs 8 held-out filename
templates, 4 attempts each, at concurrency 8. Each episode has a 3-minute time
limit and recursion limit 80. Workers have no model and need no GPU.

For an already running policy server:

```bash
PYTHONPATH=src:. python -m evals.synth.run --split eval --tasks 8 -n 4 \
  --output artifacts/evals/synth/stock-qwen4b
```

Use `--split train --tasks 16` to inspect training templates separately.
`summary.json` reports native score, pass@1, pass@4, pass^4, correct-and-parallel
rate and copies. `manifest.json` updates after each episode. Review baseline
headroom before starting OPD; an already perfect baseline cannot demonstrate
learning on this task. A script launching a known plan only checks the harness.
