# WideSearch Evaluation

Run `python -m evals.widesearch.run <run-directory> --source <dataset-directory>`
after running agents with `gyms.widesearch.run`. The default source is
`artifacts/data/widesearch/source`. Scoring can be repeated without model-agent
calls; successfully evaluated episodes are reused and evaluator errors retried.

We reuse the pinned upstream Markdown parser, key alignment prompts, per-column
criteria, row/item F1 and table success calculation. The only model adapter sends
upstream judge prompts through the repository's Flash Next non-thinking profile
(the same judge profile as BrowseComp), instead of upstream's GPT-4.1 default.
This judge choice and the fixed BrowseComp-Plus corpus differ from upstream's
published setup.

Results are separate from raw agent artifacts under `<run-directory>/evaluation`:
`provenance.json`, `summary.json`, and per-episode `result.json`, `cells.csv`, and
`judge_calls.json` containing judge prompts and responses. Gold data only enters
this evaluation process. Malformed judge JSON and upstream evaluator exceptions
are recorded as evaluation errors, not successful zero-score evaluations.

For success, row F1 and item F1, `avg` averages all scheduled attempts and `max`
averages the best repeat for each task. For binary success, `max` is empirical
pass@N. Missing attempts and agent/evaluation failures contribute zero to these
summaries; counts remain visible. Intermediate summaries therefore have a fixed
denominator and are not final results. Inspect raw results and traces alongside
aggregate scores.
