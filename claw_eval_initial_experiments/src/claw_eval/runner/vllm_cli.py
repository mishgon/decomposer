"""CLI helpers for vLLM readiness polling."""

from __future__ import annotations

import argparse

from ..config import Config, RoleName, collect_vllm_wait_targets
from .vllm_readiness import wait_for_vllm_targets


def maybe_wait_for_vllm(
    cfg: Config,
    args: argparse.Namespace,
    *,
    roles: list[RoleName],
    cli_overrides: dict[RoleName, str | None] | None = None,
    include_judge: bool = True,
) -> None:
    """Poll vLLM /v1/models before starting a run when configured."""
    if getattr(args, "no_vllm_wait", False):
        print("[vllm-wait] skipped (--no-vllm-wait)")
        return
    if cfg.provider_mode != "vllm":
        return

    targets = collect_vllm_wait_targets(
        cfg,
        roles=roles,
        cli_overrides=cli_overrides,
        include_judge=include_judge and not getattr(args, "no_judge", False),
    )
    if not targets:
        return

    parts = [f"{model_id}@{base_url}" for base_url, _key, model_id in targets]
    print(f"[provider] vllm — waiting for: {', '.join(parts)}")
    wait_for_vllm_targets(
        targets,
        timeout_s=cfg.vllm.ready_timeout_s,
        poll_s=cfg.vllm.ready_poll_s,
        check_health=cfg.vllm.check_health,
    )
