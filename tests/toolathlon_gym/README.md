# Environment integration checks

These probes exercise isolated task fixtures and MCP servers. They do not train
models or send inference requests. Run from the repository root with its Python
environment and `PYTHONPATH=src:.`.

```bash
python -m tests.toolathlon_gym.check_environment --output /path/to/new/probe-dir
python -m tests.toolathlon_gym.check_evaluators --split /path/to/split.json --output /path/to/new/probe-dir
python -m tests.toolathlon_gym.check_refunds --image IMAGE --output /path/to/new/probe-dir
```

Each probe requires a new output directory and retains diagnostic artifacts.
Refund checks use one-order pages to avoid the MCP response truncation limit.
They compare the fixture's refund IDs, reasons and totals with both order-list
and single-order responses, without changing fixture data.

The original RL image returned zero embedded refunds despite 13 records in
`wc.refunds`; the patched image passed the same probe on 2026-09-10.
The read-path patch is in `gyms/toolathlon_gym/patches/postgres-services.patch`.
It exposes the standard `id`, `reason`, and negative `total` order-refund summary,
as documented in the [WooCommerce order API](https://developer.woocommerce.com/docs/apis/rest-api/v3/orders/).
This is an environment correctness test, not evidence of RL reward improvement.
