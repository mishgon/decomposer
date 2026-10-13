from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


def completion_payload(payload, tokenizer, max_model_len):
    if payload.get("stream") or payload.get("tool_choice", "auto") not in ("auto", "none"):
        raise ValueError("Only non-streaming auto/none tool choice is supported")
    messages = json.loads(json.dumps(payload["messages"]))
    for message in messages:
        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        for call in message.get("tool_calls") or []:
            arguments = call["function"].get("arguments")
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments or "{}")
    template_options = {"enable_thinking": True, **payload.get("chat_template_kwargs", {})}
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=payload.get("tools"),
        tokenize=True,
        add_generation_prompt=True,
        **template_options,
    )
    result = {
        key: value for key, value in payload.items()
        if key not in {
            "messages", "tools", "tool_choice", "parallel_tool_calls",
            "chat_template_kwargs", "max_completion_tokens",
        }
    }
    result.update(
        prompt=prompt["input_ids"],
        skip_special_tokens=False,
        max_tokens=(payload.get("max_completion_tokens") or payload.get("max_tokens")
                    or max_model_len - len(prompt["input_ids"])),
    )
    return result


def chat_response(response, payload, tokenizer):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    request = ChatCompletionRequest(**json.loads(json.dumps(payload)))
    # vLLM 0.19 treats anyOf as JSON, even for phone numbers typed string | null.
    # Give only the response parser a type hint; keep the model's schema intact.
    for tool in request.tools or []:
        for parameter in tool.function.parameters.get("properties", {}).values():
            if {branch.get("type") for branch in parameter.get("anyOf", [])} == {"string", "null"}:
                parameter["type"] = "string"
    choices = []
    for choice in response["choices"]:
        reasoning, content = Qwen3ReasoningParser(tokenizer).extract_reasoning(
            choice["text"], request
        )
        message = {"role": "assistant", "content": content, "reasoning": reasoning}
        finish_reason = choice["finish_reason"]
        if payload.get("tools") and payload.get("tool_choice", "auto") != "none":
            parsed = Qwen3CoderToolParser(tokenizer).extract_tool_calls(content or "", request)
            if parsed.tools_called:
                message.update(
                    content=parsed.content,
                    tool_calls=[call.model_dump() for call in parsed.tool_calls],
                )
                if finish_reason != "length":
                    finish_reason = "tool_calls"
        choices.append({"index": choice["index"], "message": message, "finish_reason": finish_reason})
    return {
        "id": response["id"], "object": "chat.completion",
        "created": response["created"], "model": response["model"],
        "choices": choices, "usage": response["usage"],
    }


def create_app(upstream, tokenizer):
    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(base_url=upstream, timeout=600, trust_env=False) as client:
            app.state.client = client
            response = await client.get("/v1/models")
            response.raise_for_status()
            app.state.max_model_len = response.json()["data"][0]["max_model_len"]
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"tool_parser": "qwen3_coder"}

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        payload = await request.json()
        try:
            completion = await asyncio.to_thread(
                completion_payload, payload, tokenizer, app.state.max_model_len
            )
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        response = await app.state.client.post("/v1/completions", json=completion)
        if response.is_error:
            return JSONResponse(response.json(), status_code=response.status_code)
        return await asyncio.to_thread(chat_response, response.json(), payload, tokenizer)

    return app


def main():
    parser = argparse.ArgumentParser(description="GAIA2 client-side parsing for vLLM 0.19.0.")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--upstream", default="http://127.0.0.1:8024")
    parser.add_argument("--port", type=int, default=18025)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    uvicorn.run(create_app(args.upstream, tokenizer), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
