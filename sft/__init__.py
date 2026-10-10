"""Shared Decomposer SFT code: dataset builds (schema, builder, adapters) and training.

Gym-specific trace preparation lives in `sft/<gym>/`. Import modules directly
(`sft.builder`, `sft.train`, ...): the package stays import-free so launchers that
run in minimal environments (for example `sft.run_train_jobs`) do not pull in the
dataset stack.
"""
