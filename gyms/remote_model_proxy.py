"""Credential-isolating proxy for remote OpenAI-compatible model servers.

Some Qwen deployments render tool calls with the model's native XML syntax
without enabling the serving-layer tool parser.  ``qwen3_xml`` normalization
converts those response text blocks into Responses API function-call items so
the caller sees the same interface it would receive from a parser-enabled
server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import uuid
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
_TOOL_BLOCK_RE = re.compile(
    r"<\s*tool_call\s*>(.*?)<\s*/\s*tool_call\s*>", re.DOTALL
)
_FUNCTION_RE = re.compile(
    r"<\s*function\s*=\s*([^>]+?)\s*>(.*?)<\s*/\s*function\s*>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]+?)\s*>(.*?)<\s*/\s*parameter\s*>",
    re.DOTALL,
)


class ToolCallParseError(ValueError):
    """The upstream emitted tool-like text that could not be normalized."""


def _merge_extra_body(
    payload: Mapping[str, Any], extra_body: Mapping[str, Any]
) -> dict[str, Any]:
    merged = dict(payload)
    merged.update(extra_body)
    return merged


def _retryable_status(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUSES


def _without_matches(value: str, matches: Sequence[re.Match[str]]) -> str:
    pieces: list[str] = []
    cursor = 0
    for match in matches:
        pieces.append(value[cursor : match.start()])
        cursor = match.end()
    pieces.append(value[cursor:])
    return "".join(pieces)


def _tool_schemas(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for item in payload.get("tools") or []:
        if not isinstance(item, Mapping) or item.get("type") != "function":
            continue
        name = item.get("name")
        parameters = item.get("parameters")
        if isinstance(name, str) and isinstance(parameters, Mapping):
            schemas[name] = dict(parameters)
    return schemas


def _coerce_argument(raw: str, schema: Mapping[str, Any]) -> Any:
    expected = schema.get("type")
    if isinstance(expected, list):
        expected = next((item for item in expected if item != "null"), None)
    value = raw.strip()
    if expected == "string" or expected is None:
        return value
    if expected == "integer":
        return int(value)
    if expected == "number":
        return float(value)
    if expected == "boolean":
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise ValueError(f"expected a boolean, found {value!r}")
    if expected == "null":
        if value.lower() == "null":
            return None
        raise ValueError(f"expected null, found {value!r}")
    if expected in {"array", "object"}:
        parsed = json.loads(value)
        if expected == "array" and not isinstance(parsed, list):
            raise ValueError("expected an array")
        if expected == "object" and not isinstance(parsed, dict):
            raise ValueError("expected an object")
        return parsed
    return value


def _parse_function(
    match: re.Match[str], schemas: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    name = match.group(1).strip()
    if name not in schemas:
        raise ToolCallParseError(f"Qwen emitted unknown tool {name!r}")
    body = match.group(2)
    parameter_matches = list(_PARAMETER_RE.finditer(body))
    if _without_matches(body, parameter_matches).strip():
        raise ToolCallParseError(f"Malformed Qwen arguments for tool {name!r}")

    parameter_schema = schemas[name]
    properties = parameter_schema.get("properties") or {}
    if not isinstance(properties, Mapping):
        properties = {}
    arguments: dict[str, Any] = {}
    for parameter_match in parameter_matches:
        parameter_name = parameter_match.group(1).strip()
        if parameter_name in arguments:
            raise ToolCallParseError(
                f"Qwen emitted duplicate argument {parameter_name!r} for {name!r}"
            )
        schema = properties.get(parameter_name)
        if not isinstance(schema, Mapping):
            if parameter_schema.get("additionalProperties") is False:
                raise ToolCallParseError(
                    f"Qwen emitted unknown argument {parameter_name!r} for {name!r}"
                )
            schema = {}
        try:
            arguments[parameter_name] = _coerce_argument(
                parameter_match.group(2), schema
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ToolCallParseError(
                f"Invalid Qwen argument {parameter_name!r} for {name!r}: {error}"
            ) from error

    required = parameter_schema.get("required") or []
    missing = [item for item in required if item not in arguments]
    if missing:
        raise ToolCallParseError(
            f"Qwen omitted required arguments for {name!r}: {missing}"
        )
    identifier = uuid.uuid4().hex
    return {
        "type": "function_call",
        "id": f"fc_{identifier}",
        "call_id": f"call_{identifier}",
        "name": name,
        "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        "status": "completed",
    }


def parse_qwen3_xml_tool_calls(
    text: str, schemas: Mapping[str, dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Return residual user-visible text and normalized function calls."""

    markers_present = "<tool_call" in text or "<function=" in text
    blocks = list(_TOOL_BLOCK_RE.finditer(text))
    if not blocks:
        if markers_present:
            raise ToolCallParseError("Malformed Qwen tool-call XML")
        return text, []

    calls: list[dict[str, Any]] = []
    for block in blocks:
        body = block.group(1)
        functions = list(_FUNCTION_RE.finditer(body))
        if not functions or _without_matches(body, functions).strip():
            raise ToolCallParseError("Malformed Qwen function-call XML")
        calls.extend(_parse_function(match, schemas) for match in functions)

    residual = _without_matches(text, blocks)
    if "<tool_call" in residual or "<function=" in residual:
        raise ToolCallParseError("Unparsed Qwen tool-call XML remains")
    return residual, calls


def normalize_qwen3_xml_response(
    response: Mapping[str, Any], request_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize non-streaming Responses API output from a Qwen deployment."""

    normalized = dict(response)
    output = response.get("output")
    if not isinstance(output, list):
        raise ToolCallParseError("Responses API payload has no output list")
    schemas = _tool_schemas(request_payload)
    rewritten: list[Any] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            rewritten.append(item)
            continue
        content = item.get("content")
        if not isinstance(content, list):
            rewritten.append(item)
            continue
        rewritten_content: list[Any] = []
        calls: list[dict[str, Any]] = []
        for part in content:
            if (
                isinstance(part, Mapping)
                and part.get("type") in {"output_text", "text"}
                and isinstance(part.get("text"), str)
            ):
                residual, parsed = parse_qwen3_xml_tool_calls(part["text"], schemas)
                calls.extend(parsed)
                if residual.strip():
                    updated = dict(part)
                    updated["text"] = residual
                    rewritten_content.append(updated)
            else:
                rewritten_content.append(part)
        if rewritten_content:
            updated_item = dict(item)
            updated_item["content"] = rewritten_content
            rewritten.append(updated_item)
        rewritten.extend(calls)
    normalized["output"] = rewritten
    return normalized


def create_app(
    *,
    upstream_url: str,
    api_key: str,
    extra_body: Mapping[str, Any] | None = None,
    response_tool_parser: str | None = None,
    timeout_seconds: float = 3300,
    max_retries: int = 2,
    verify_tls: bool = True,
) -> FastAPI:
    if not upstream_url:
        raise ValueError("upstream_url must not be empty")
    if not api_key:
        raise ValueError("api_key must not be empty")
    if max_retries < 0:
        raise ValueError("max_retries must not be negative")
    if response_tool_parser not in (None, "qwen3_xml"):
        raise ValueError(f"Unknown response tool parser: {response_tool_parser!r}")

    app = FastAPI(title="Remote Model Proxy", version="1")
    upstream = upstream_url.rstrip("/")
    injected = dict(extra_body or {})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route("/v1/{path:path}", methods=("GET", "POST"))
    async def forward(path: str, request: Request) -> Response:
        payload: dict[str, Any] | None = None
        if request.method == "POST":
            try:
                value = await request.json()
            except json.JSONDecodeError as error:
                raise HTTPException(status_code=400, detail="Expected JSON body") from error
            if not isinstance(value, dict):
                raise HTTPException(status_code=400, detail="Expected JSON object")
            payload = _merge_extra_body(value, injected)
            if response_tool_parser and payload.get("stream") is True:
                raise HTTPException(
                    status_code=400,
                    detail="Response tool normalization requires non-streaming output",
                )

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
            verify=verify_tls,
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
                    if _retryable_status(response.status_code) and attempt < max_retries:
                        await asyncio.sleep(2**attempt)
                        continue
                    content = response.content
                    if (
                        response_tool_parser == "qwen3_xml"
                        and request.method == "POST"
                        and path == "responses"
                        and response.status_code < 400
                        and payload is not None
                    ):
                        try:
                            parsed_response = response.json()
                            content = json.dumps(
                                normalize_qwen3_xml_response(parsed_response, payload),
                                ensure_ascii=False,
                            ).encode()
                        except (json.JSONDecodeError, ToolCallParseError) as error:
                            raise HTTPException(status_code=502, detail=str(error)) from error
                    response_headers = {
                        name: value
                        for name, value in response.headers.items()
                        if name.lower() in _FORWARDED_RESPONSE_HEADERS
                    }
                    return Response(
                        content=content,
                        status_code=response.status_code,
                        headers=response_headers,
                    )
                await asyncio.sleep(2**attempt)

        error_name = type(last_error).__name__ if last_error is not None else "unknown"
        raise HTTPException(
            status_code=502,
            detail=f"Remote model transport failed after retries ({error_name})",
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream-url")
    parser.add_argument("--upstream-url-env")
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--extra-body-json", default="{}")
    parser.add_argument("--response-tool-parser", choices=("qwen3_xml",))
    parser.add_argument("--timeout-seconds", type=float, default=3300)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--no-verify-tls", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.upstream_url) == bool(args.upstream_url_env):
        raise ValueError("Configure exactly one of --upstream-url or --upstream-url-env")
    upstream_url = args.upstream_url or os.environ.get(args.upstream_url_env, "")
    if not upstream_url:
        raise RuntimeError(f"{args.upstream_url_env} is required")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is required")
    extra_body = json.loads(args.extra_body_json)
    if not isinstance(extra_body, dict):
        raise TypeError("--extra-body-json must decode to an object")
    app = create_app(
        upstream_url=upstream_url,
        api_key=api_key,
        extra_body=extra_body,
        response_tool_parser=args.response_tool_parser,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        verify_tls=not args.no_verify_tls,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
