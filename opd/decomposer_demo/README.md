# Optional Demo Validation


This benchmark is a standalone demo regression check, not the learning metric
for the current OPD experiment. `stock-qwen4b-main-synth-smoke-20261030` uses only
synthetic validation. Demo tasks remain excluded from training. Its 20-task score
measures the complete demo stack, including hosted Flash Next frontend workers,
so it does not isolate the trainable coordinator's improvement.

The reusable OPD adapter remains available for explicit future experiments.
To opt in, add `--demo-validation` to `python -m opd.synth.run ...`.
Preparation leaves synthetic training data unchanged and appends 40 validation
rows: the same 20 tasks in EN and RU. Sources `decomposer-demo/en` and
`decomposer-demo/ru` keep language metrics separate. The default remains synthetic
validation only. Current MLSpace experiments do not pass this flag.

`opd.decomposer_demo.agent_loop.DemoAgentLoop` uses the existing student token
bridge for the coordinator, the demo's hosted worker-model assignments, and the same
in-process episode and native reward as standalone evaluation. `opd/setup.sh`
installs the CSS parser through the pinned OPD requirements lock. No teacher
reward or generated-code execution is needed for these validation episodes.

For consumers that already have authoritative rollout state, the optional
`opd.decomposer_demo.reward.compute_score` adapter exposes the veRL custom reward
signature. Supply the task ID as `ground_truth`, and harness state in
`extra_info={'final_state': state, 'language': 'en'}`. It delegates to the same
`gyms.decomposer_demo.grade` function and rejects missing state, unknown IDs,
non-validation sources, and inconsistent language labels. Never construct this
state from model-written claims of success. For estate tasks, the harness alone
extracts submitted source URLs from the final answer and combines them with
recorded search/fetch evidence.
