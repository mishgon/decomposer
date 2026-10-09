"""Stop collection after an overlength teacher response has been checkpointed."""
import asyncio
from typing import NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage

from sft.filtering import StudentSequenceFilter


class SequenceState(AgentState):
    sft_format: NotRequired[dict]
    sequence_length: NotRequired[int]
    stop_reason: NotRequired[str]


class StudentSequenceLimit(AgentMiddleware):
    state_schema = SequenceState

    def __init__(self, tokenizer):
        self.filter = StudentSequenceFilter(tokenizer)

    def measure(self, request, response):
        format = request.state.get('sft_format') or self.filter.format(request.system_message, request.tools)
        check = self.filter.check([*request.messages, *response.result], format)
        message = next(m for m in reversed(response.result) if isinstance(m, AIMessage))
        message.additional_kwargs['sft_sequence'] = {'tokens': check.tokens, 'limit': check.limit}
        if 'sft_format' not in request.state:
            message.additional_kwargs['sft_format'] = format
        return response

    def wrap_model_call(self, request, handler):
        return self.measure(request, handler(request))

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        return await asyncio.to_thread(self.measure, request, response)

    @hook_config(can_jump_to=['end'])
    def after_model(self, state, runtime):
        message = next(m for m in reversed(state['messages']) if isinstance(m, AIMessage))
        check = message.additional_kwargs['sft_sequence']
        update = {'sequence_length': check['tokens']}
        if 'sft_format' in message.additional_kwargs:
            update['sft_format'] = message.additional_kwargs['sft_format']
        if check['tokens'] > check['limit']:
            update.update(stop_reason='sequence_limit', jump_to='end')
        return update

    @hook_config(can_jump_to=['end'])
    async def aafter_model(self, state, runtime):
        return self.after_model(state, runtime)
