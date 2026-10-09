import asyncio
import os
import pytest


def test_collection_tokenizer_loads_outside_event_loop(monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys
    import threading
    from types import SimpleNamespace

    main_thread = threading.get_ident()
    def load(path, *, local_files_only):
        assert threading.get_ident() != main_thread
        assert path == '/opt/sft-tokenizer' and local_files_only
        return Tokenizer(1)

    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load)))
    monkeypatch.setitem(sys.modules, 'agents', SimpleNamespace(decomposer=lambda **kw: kw))
    path = Path(__file__).resolve().parents[1] / 'sft/toolathlon_gym/agents.py'
    spec = importlib.util.spec_from_file_location('collection_agents_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    graph = asyncio.run(module.decomposer())
    assert isinstance(graph['middleware'][0], StudentSequenceLimit)

from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, message_to_dict
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from sft.filtering import StudentSequenceFilter, student_messages
from sft.sequence_limit import StudentSequenceLimit


class Tokenizer:
    name_or_path = 'test-tokenizer'
    chat_template = 'test-template'

    def __init__(self, count):
        self.count = count

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return range(self.count)


def test_skip_is_terminal_but_never_successful(tmp_path):
    import json
    from sft.toolathlon_gym.collection import count_statuses, next_attempt
    from sft.toolathlon_gym.scheduler import TERMINAL_EPISODE_STATUSES, load_launch_outcome
    from gyms.wideseek.metrics import qualifies

    result = tmp_path / 'evaluation.json'
    result.write_text(json.dumps({'pass': True, 'stop_reason': 'sequence_limit'}))
    assert not load_launch_outcome('task', str(result)).qualifies(.9)
    assert 'skipped' in TERMINAL_EPISODE_STATUSES
    assert not qualifies({'status': 'skipped', 'evaluation': {'status': 'scored', 'score': 1.}}, .9)
    saved = {'attempt': 1, 'status': 'skipped', 'stop_reason': 'sequence_limit', 'error': None}
    path = tmp_path / 'attempts/task/rep-001/attempt-001'
    path.mkdir(parents=True)
    (path / 'attempt.json').write_text(json.dumps(saved))
    episode = {'task': 'task', 'repetition': 1, 'attempts': [], 'status': 'pending'}
    assert next_attempt(tmp_path, episode) == (2, True)
    assert episode['status'] == 'skipped'
    counts = count_statuses({'episodes': [episode]})
    assert counts['skipped'] == counts['total'] == 1
    assert counts['failed'] == counts['completed'] == 0


class Policy(BaseChatModel):
    @property
    def _llm_type(self):
        return 'test'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage('raw answer',
            additional_kwargs={'reasoning_content': 'private reasoning'},
            tool_calls=[{'name': 'touch', 'args': {}, 'id': 'call1', 'type': 'tool_call'}]))])


def test_batch_and_live_agree_and_preserve_raw():
    tokenizer = Tokenizer(32769)
    middleware = StudentSequenceLimit(tokenizer)
    request = ModelRequest(model=Policy(), messages=[HumanMessage('task')],
                           system_message=SystemMessage('system'), tools=[], state={})
    answer = AIMessage('answer', additional_kwargs={'reasoning_content': 'reasoning'})
    response = middleware.measure(request, ModelResponse(result=[answer]))
    state = {'messages': [*request.messages, *response.result]}
    update = middleware.after_model(state, None)
    assert update['jump_to'] == 'end'
    assert update['stop_reason'] == 'sequence_limit'
    assert answer.additional_kwargs['reasoning_content'] == 'reasoning'
    raw = [message_to_dict(m) for m in state['messages']]
    check = middleware.filter.check(raw, update['sft_format'])
    assert check.tokens == update['sequence_length'] == 32769
    assert tokenizer.messages[0] == {'role': 'system', 'content': 'system'}
    assert 'reasoning_content' not in tokenizer.messages[-1]
    assert tokenizer.kwargs['enable_thinking'] is False


def test_boundary_is_strictly_greater():
    tokenizer = Tokenizer(32768)
    middleware = StudentSequenceLimit(tokenizer)
    request = ModelRequest(model=Policy(), messages=[HumanMessage('task')], state={}, tools=[])
    response = middleware.measure(request, ModelResponse(result=[AIMessage('answer')]))
    assert 'jump_to' not in middleware.after_model({'messages': response.result}, None)


@pytest.mark.skipif(not os.environ.get('SFT_TOKENIZER'), reason='Requires local student tokenizer')
def test_real_student_template_and_raw_serialization():
    from transformers import AutoTokenizer
    from langchain_core.messages import ToolMessage
    tokenizer = AutoTokenizer.from_pretrained(os.environ['SFT_TOKENIZER'], local_files_only=True)
    checker = StudentSequenceFilter(tokenizer)
    tools = [{'type': 'function', 'function': {'name': 'lookup', 'description': 'Search',
              'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}}}]
    messages = [HumanMessage('Найди 東京'), AIMessage('', tool_calls=[{
        'id': 'c1', 'name': 'lookup', 'args': {'query': '東京'}, 'type': 'tool_call'}]),
        ToolMessage('result', tool_call_id='c1'),
        AIMessage('answer', additional_kwargs={'reasoning_content': 'hidden ' * 10000})]
    format = checker.format(SystemMessage('system'), tools)
    count = checker.check(messages, format).tokens
    assert count > 50
    assert count == checker.check([message_to_dict(m) for m in messages], format).tokens
    assert count == checker.check([m.model_dump(mode='json') for m in messages], format).tokens
    plain = student_messages(messages, SystemMessage('system'))
    assert count == len(tokenizer.apply_chat_template(plain, tools=tools, tokenize=True,
        return_dict=False, add_generation_prompt=False, enable_thinking=False, preserve_thinking=False))
    messages[-1] = AIMessage('answer')
    assert count == checker.check(messages, format).tokens


@pytest.mark.parametrize('decomposer', [False, True])
def test_graph_saves_overlength_turn_without_executing_tools(decomposer):
    executed = []
    def touch() -> str:
        """Record execution."""
        executed.append(True)
        return 'done'
    agent = create_agent(Policy(), tools=[touch],
        middleware=[StudentSequenceLimit(Tokenizer(32769))], checkpointer=InMemorySaver())
    if decomposer:
        from decomposer.core import create_decomposer_agent
        agent = create_decomposer_agent(decomposer_model=Policy(), agent_types=[{
            'agent_type_id': 'worker', 'description': 'worker', 'assistant_id': 'worker',
            'url': 'http://127.0.0.1:1'}],
            middleware=[StudentSequenceLimit(Tokenizer(32769))], checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': 'test'}}
    result = asyncio.run(agent.ainvoke({'messages': [HumanMessage('task')]}, config))
    assert result['stop_reason'] == 'sequence_limit'
    assert result['messages'][-1].content == 'raw answer'
    assert result['messages'][-1].tool_calls
    assert not executed
    assert agent.get_state(config).values['sequence_length'] == 32769
