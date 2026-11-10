# SFT 1.4.0 Evaluation Campaign

Run `python -m evals.sft140` from the primary h200 checkout with credentials in the process environment. The queue needs `--output-dir`, `--browsecomp-data` (the existing 100-task selection), and `--old-browsecomp-run` (the preserved 60+60-second run).

The schedule contains 25 jobs: twelve one-task smoke runs, a five-task BrowseComp pilot with three repeats, and twelve full runs. Full runs cover five configurations on BrowseComp and English WideSearch, followed by SFT on Gaia2 Execution and Search. BrowseComp and Gaia2 use concurrency 2; WideSearch uses concurrency 1. The queue uses the existing model registry and benchmark runners.

The server process must be detached from SSH and receive its credentials before launch. `queue.json` records the commit, selection hash, commands, current child PID, progress, and completed jobs. Each job keeps a runner log and preflight evidence; benchmark runners retain raw traces. `pilot-comparison.json` compares the same five tasks with the old run. Model servers and retrieval remain running when the queue exits.

Rerunning the same command resumes the queue only with the same commit, selection, and schedule. BrowseComp and WideSearch retry technical failures and preserve previous attempts; saved answers with judge failures can be rescored. A partial Gaia2 run pauses for inspection because its runner does not support resumption. The queue preserves its directory and never replaces it automatically.

The queue stops if preflight fails, a runner reports infrastructure trouble, or scoring remains incomplete. It does not increase concurrency or change task selection automatically. Old 60+60-second results remain separate from new 180+180-second results.
