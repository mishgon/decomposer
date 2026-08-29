"""Compatibility entry point for the shared remote-model proxy."""

from gyms.remote_model_proxy import (  # noqa: F401
    ToolCallParseError,
    _merge_extra_body,
    _retryable_status,
    build_parser,
    create_app,
    main,
    normalize_qwen3_xml_response,
    parse_qwen3_xml_tool_calls,
)
from gyms.remote_model_proxy import asyncio, httpx  # noqa: F401


if __name__ == "__main__":
    raise SystemExit(main())
