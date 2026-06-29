"""Wait for vLLM models to become available before starting agent runs."""

from __future__ import annotations

import time
from urllib.parse import urlparse

import httpx


class VllmReadinessTimeoutError(TimeoutError):
    """Raised when a vLLM model does not become ready within the timeout."""


def normalize_vllm_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def _health_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc or parsed.path.split("/")[0]
    return f"{scheme}://{netloc}/health"


def _model_listed(model_id: str, models_payload: dict) -> bool:
    data = models_payload.get("data", [])
    if not isinstance(data, list):
        return False
    ids = {item.get("id") for item in data if isinstance(item, dict)}
    if model_id in ids:
        return True
    # Some servers expose a suffix or alias; allow substring match on id tail.
    tail = model_id.rsplit("/", 1)[-1]
    return any(tail in (mid or "") for mid in ids)


def wait_for_vllm_model(
    *,
    base_url: str,
    model_id: str,
    api_key: str = "unused",
    timeout_s: int = 600,
    poll_s: float = 5.0,
    check_health: bool = True,
) -> None:
    """Poll vLLM until *model_id* appears in GET /v1/models."""
    base_url = normalize_vllm_base_url(base_url)
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.monotonic() + timeout_s
    attempt = 0

    while time.monotonic() < deadline:
        attempt += 1
        try:
            if check_health:
                health_resp = httpx.get(_health_url(base_url), timeout=10.0)
                if health_resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        "health check failed",
                        request=health_resp.request,
                        response=health_resp,
                    )

            resp = httpx.get(f"{base_url}/models", headers=headers, timeout=30.0)
            if resp.status_code == 200:
                payload = resp.json()
                if _model_listed(model_id, payload):
                    print(f"[vllm-wait] ready: {model_id} on {base_url}")
                    return
            elif resp.status_code not in (404, 503):
                resp.raise_for_status()
        except Exception as exc:
            if attempt == 1 or attempt % 6 == 0:
                print(
                    f"[vllm-wait] waiting for {model_id} on {base_url} "
                    f"(attempt {attempt}, {type(exc).__name__}: {exc})"
                )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_s, remaining))

    raise VllmReadinessTimeoutError(
        f"Timed out after {timeout_s}s waiting for model {model_id!r} on {base_url}. "
        "Start vLLM with scripts/vllm/serve_single.sh or serve_decomposer_pair.sh."
    )


def wait_for_vllm_targets(
    targets: list[tuple[str, str, str]],
    *,
    timeout_s: int = 600,
    poll_s: float = 5.0,
    check_health: bool = True,
) -> None:
    """Wait for each unique (base_url, api_key, model_id) target."""
    for base_url, api_key, model_id in targets:
        wait_for_vllm_model(
            base_url=base_url,
            model_id=model_id,
            api_key=api_key,
            timeout_s=timeout_s,
            poll_s=poll_s,
            check_health=check_health,
        )
