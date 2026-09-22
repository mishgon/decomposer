"""LangChain model bridge: veRL generates tokens; existing Decomposer owns tools."""

import asyncio
import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from decomposer.chat_vllm import ChatVLLM


class RolloutBudgetExceeded(Exception):
    pass


class PolicyTokens:
    def __init__(self, tokenizer, generate, sampling_params, *, prompt_budget, response_budget, log_path=None):
        self.tokenizer, self.generate = tokenizer, generate
        self.sampling_params = dict(sampling_params)
        self.prompt_budget, self.response_budget = prompt_budget, response_budget
        self.prompt_ids, self.response_ids, self.mask, self.logprobs = [], [], [], []
        self.turns, self.extra, self.calls = 0, {}, []
        self.message_count = 0
        self.log_path = log_path

    def render(self, messages, **kwargs):
        return list(self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, enable_thinking=False, **kwargs))

    def observe(self, messages, tools):
        if not self.turns:
            self.prompt_ids = self.render(messages, tools=tools, add_generation_prompt=True)
            if len(self.prompt_ids) > self.prompt_budget:
                raise ValueError("Initial prompt exceeds training prompt budget")
        else:
            # The previous assistant's raw token IDs are already in response_ids.
            delta = messages[self.message_count + 1:]
            if not delta or any(message["role"] == "assistant" for message in delta):
                raise ValueError("Expected only new observations after the generated assistant")
            stub = [{"role": "user", "content": "stub"}, {"role": "assistant", "content": "x"}]
            wrapped = self.render(stub + delta, add_generation_prompt=True)
            # EOS generation omits the template's newline after the closing token.
            eos = self.tokenizer.eos_token_id
            if self.response_ids[-1] != eos:
                raise ValueError("Previous assistant did not terminate with EOS")
            # Locate the stub's two closing tokens in THIS rendering. A new
            # user message makes Qwen remove the stub's empty reasoning scaffold,
            # so its length cannot be measured from a separate rendering.
            closes = [i for i, token in enumerate(wrapped) if token == eos]
            if len(closes) < 2:
                raise ValueError("Could not locate observation boundary")
            observation = wrapped[closes[1] + 1:]
            if len(self.response_ids) + len(observation) >= self.response_budget:
                raise RolloutBudgetExceeded("Observation exceeds remaining response budget")
            self.response_ids.extend(observation)
            self.mask.extend([0] * len(observation))
            self.logprobs.extend([0.0] * len(observation))
        self.message_count = len(messages)

    async def respond(self, messages, tools):
        self.observe(messages, tools)
        remaining = self.response_budget - len(self.response_ids)
        if remaining <= 0:
            raise RolloutBudgetExceeded("Response budget exhausted")
        try:
            output = await asyncio.wait_for(self.generate(
                self.prompt_ids + self.response_ids,
                {**self.sampling_params, "max_tokens": remaining}), timeout=300)
        except TimeoutError as error:
            raise RuntimeError("Policy inference request timed out") from error
        ids, probs = list(output.token_ids), output.log_probs
        if not ids or probs is None or len(probs) != len(ids) or len(ids) > remaining:
            raise ValueError("Invalid policy token/logprob alignment")
        self.response_ids.extend(ids)
        self.mask.extend([1] * len(ids))
        self.logprobs.extend(probs)
        self.turns += 1
        for key, value in (output.extra_fields or {}).items():
            if value is None:
                continue
            if key == "min_global_steps":
                self.extra[key] = min(self.extra.get(key, value), value)
            elif key == "max_global_steps":
                self.extra[key] = max(self.extra.get(key, value), value)
            else:
                self.extra[key] = value
        raw = self.tokenizer.decode(ids, skip_special_tokens=False)
        self.calls.append({"text": raw, "token_ids": ids, "log_probs": probs,
                           "extra_fields": output.extra_fields})
        if self.log_path is not None:
            with self.log_path.open("a") as stream:
                stream.write(json.dumps(self.calls[-1]) + "\n")
        if ids[-1] != self.tokenizer.eos_token_id:
            raise RolloutBudgetExceeded("Generation exhausted budget without EOS")
        content = raw.removesuffix(self.tokenizer.eos_token)
        try:
            content, calls = ChatVLLM._parse_qwen_xml(content)
        except ValueError:
            # Malformed policy output is an agent outcome, not a broken server.
            calls = []
        return AIMessage(content=content, tool_calls=calls)


class VerlChatModel(BaseChatModel):
    tokens: Any = Field(exclude=True)

    @property
    def _llm_type(self):
        return "verl-token-policy"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

    def _generate(self, *args, **kwargs):
        raise NotImplementedError("RL policy uses async generation only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        message = await self.tokens.respond(convert_to_openai_messages(messages), kwargs["tools"])
        return ChatResult(generations=[ChatGeneration(message=message)])
