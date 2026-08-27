"""Credential-isolating OpenAI-compatible proxy for Gaia2 policy requests."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

_RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
_FORWARDED_RESPONSE_HEADERS = {
    "content-type",
    "openrouter-processing-time",
    "x-request-id",
}


def _merge_extra_body(
    payload: Mapping[str, Any], extra_body: Mapping[str, Any]
) -> dict[str, Any]:
    merged = dict(payload)
    merged.update(extra_body)
    return merged


def _retryable_status(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUSES


def create_app(
    *,
    upstream_url: str,
    api_key: str,
    extra_body: Mapping[str, Any] | None = None,
    timeout_seconds: float = 3300,
    max_retries: int = 2,
) -> FastAPI:
    if not upstream_url:
        raise ValueError("upstream_url must not be empty")
    if not api_key:
        raise ValueError("api_key must not be empty")
    if max_retries < 0:
        raise ValueError("max_retries must not be negative")

    app = FastAPI(title="Gaia2 OpenRouter Policy Proxy", version="1")
    upstream = upstream_url.rstrip("/")
    injected = dict(extra_body or {})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route(
        "/v1/{path:path}",
        methods=("GET", "POST"),
    )
    async def forward(path: str, request: Request) -> Response:
        payload: dict[str, Any] | None = None
        if request.method == "POST":
            try:
                value = await request.json()
            except json.JSONDecodeError as error:
                raise HTTPException(
                    status_code=400, detail="Expected JSON body"
                ) from error
            if not isinstance(value, dict):
                raise HTTPException(status_code=400, detail="Expected JSON object")
            payload = _merge_extra_body(value, injected)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(timeout_seconds)
        last_error: httpx.TransportError | None = None
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            trust_env=True,
        ) as client:
            for attempt in range(max_retries + 1):
                try:
                    response = await client.request(
                        request.method,
                        f"{upstream}/{path}",
                        params=request.query_params,
                        headers=headers,
                        json=payload,
                    )
                except httpx.TransportError as error:
                    last_error = error
                    if attempt == max_retries:
                        break
                else:
                    if not _retryable_status(response.status_code) or (
                        attempt == max_retries
                    ):
                        response_headers = {
                            name: value
                            for name, value in response.headers.items()
                            if name.lower() in _FORWARDED_RESPONSE_HEADERS
                        }
                        return Response(
                            content=response.content,
                            status_code=response.status_code,
                            headers=response_headers,
                        )
                await asyncio.sleep(2**attempt)

        error_name = type(last_error).__name__ if last_error is not None else "unknown"
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter transport failed after retries ({error_name})",
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--extra-body-json", default="{}")
    parser.add_argument("--timeout-seconds", type=float, default=3300)
    parser.add_argument("--max-retries", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is required")
    extra_body = json.loads(args.extra_body_json)
    if not isinstance(extra_body, dict):
        raise TypeError("--extra-body-json must decode to an object")
    app = create_app(
        upstream_url=args.upstream_url,
        api_key=api_key,
        extra_body=extra_body,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
