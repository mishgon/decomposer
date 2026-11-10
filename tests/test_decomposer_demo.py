import asyncio
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from gyms.decomposer_demo import Calendar, grade, load_task
from gyms.decomposer_demo.calendar import KEEP_IDS, TASK_ID


def expected_events():
    return [e for e in load_task()['calendar']['events'] if e['id'] in KEEP_IDS]


def test_pinned_task_matches_demo_speaker_metadata():
    task = load_task()
    assert len(task['calendar']['events']) == 40
    airi_speakers = {e['id'] for e in task['calendar']['events']
                     if any(p.get('organization') == 'AIRI' and p.get('role', '').startswith('speaker')
                            for p in e.get('people', []))}
    assert airi_speakers == KEEP_IDS
    assert task['split'] == 'validation'


def test_episode_state_and_tool_results_are_isolated():
    task = load_task()
    first, second = Calendar(task), Calendar(task)
    original = deepcopy(second.events)
    visible = first.get_events('11:00', '17:00')
    visible[0]['title'] = 'tampered'
    assert first.events == original
    first.remove_event(first.events[0]['id'])
    assert second.events == original == task['calendar']['events']
    assert load_task()['calendar']['events'] == original


def test_calendar_windows_and_end_of_day():
    calendar = Calendar(load_task())
    with pytest.raises(ValueError):
        calendar.get_events('00:00', '24:00')
    with pytest.raises(ValueError):
        calendar.get_events('24:00', '24:00')
    new = calendar.create_event('Late event', '23:00', '24:00')
    assert new['end'] == '2026-09-20T00:00:00+03:00'
    assert new in calendar.get_events('23:30', '24:00')
    assert new not in calendar.get_events('22:00', '23:00')


def test_reward_rejects_missing_extra_and_modified_events():
    expected = expected_events()
    assert grade(TASK_ID, expected[::-1]) == {'score': 1.0, 'passed': True}
    assert grade(TASK_ID, expected[:-1])['score'] == 0
    assert grade(TASK_ID, expected + [load_task()['calendar']['events'][0]])['score'] == 0
    changed = deepcopy(expected)
    changed[0]['title'] = 'modified'
    assert grade(TASK_ID, changed)['score'] == 0
    assert grade(TASK_ID, expected + expected[:1])['score'] == 0


def test_calendar_task_import_without_site_packages():
    subprocess.run([sys.executable, '-S', '-c',
        'from gyms.decomposer_demo import load_task; '
        'assert load_task()["benchmark"] == "decomposer-demo"'], check=True)


def test_in_process_react_episode_and_eval_use_same_reward():
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from evals.decomposer_demo.run import summarize
    from gyms.decomposer_demo.run import episode

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    deletes = [{'name': 'remove_event', 'args': {'id': e['id']}, 'id': e['id'], 'type': 'tool_call'}
               for e in load_task()['calendar']['events'] if e['id'] not in KEEP_IDS]
    model = ToolModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'get_events', 'args': {'from': '11:00', 'to': '17:00'},
                                          'id': 'read', 'type': 'tool_call'}]),
        AIMessage(content='', tool_calls=deletes), AIMessage(content='Done')])
    result = asyncio.run(episode(TASK_ID, model, harness="react"))
    assert result['status'] == 'finished'
    assert result['score'] == summarize([result])['mean_score'] == 1
    assert grade(TASK_ID, result['final_state'])['score'] == result['score']
    assert len(result['messages']) > 2
    json.dumps(result)


def test_twenty_tasks_language_and_reference_states():
    from gyms.decomposer_demo import grade as check
    from gyms.decomposer_demo.tasks import task_ids, task_spec, CALENDAR, FRONTEND
    from gyms.decomposer_demo.estate import expected, snapshot, matches
    from gyms.decomposer_demo.frontend import seed
    ids=task_ids()
    assert len(ids)==len(set(ids))==20
    assert sum(task_spec(id)['environment']=='calendar' for id in ids)==8
    for language in ('en','ru'):
        for id in ids:
            task=task_spec(id,language)
            assert task['prompt'] != task_spec(id,'ru' if language=='en' else 'en')['prompt']
            if task['environment']=='calendar':
                keep=next(t[3] for t in CALENDAR if t[0]==id)
                state=[e for e in load_task()['calendar']['events'] if keep(e)]
            elif task['environment']=='estate':
                shortlist=expected(id)
                assert (len(shortlist)==0) == (id=='estate-no-matches')
                assert all(matches(r,task['filters']) for r in snapshot() if r['id'] in shortlist)
                rows = {r['id']: r for r in snapshot()}
                urls = [rows[id]['url'] for id in shortlist]
                state = {'answer': '\n'.join(urls) if urls else 'No matches.',
                         'submitted_urls': urls, 'fetched': shortlist, 'searches': []}
                if not shortlist:
                    state['searches'] = [{'city': task['filters']['city'], 'deal_type': task['filters']['deal'],
                        'rooms': 4, 'min_price': 1000000, 'max_price': 2000000, 'total': 0}]
            else:
                state=seed(language)
                state['/styles.css']+='\n'+'\n'.join(f'{selector} {{ {key}: {value}; }}' for (selector,key),value in task['targets'].items())
            assert check(id,state,language)['score']==1, id
    assert seed('ru')['/App.jsx']!=seed('en')['/App.jsx']
    assert seed('ru')['/ChatDemo.jsx']!=seed('en')['/ChatDemo.jsx']


def test_frontend_revision_and_preservation():
    from gyms.decomposer_demo.frontend import Frontend, worker_id, grade as check
    gym=Frontend('en')
    worker_id.set('a');gym.read_file('/styles.css')
    worker_id.set('b');gym.read_file('/styles.css');gym.write_file('/styles.css',gym.files['/styles.css']+'\n')
    worker_id.set('a')
    with pytest.raises(ValueError,match='Revision'):
        gym.write_file('/styles.css','body {}')
    with pytest.raises(ValueError):gym.write_file('/../../secret','x')
    state=gym.state();state['/api.js']='broken'
    assert check('frontend-notebook',state)['score']==0
    worker_id.set('react')


def test_estate_independent_state_and_final_submission():
    from gyms.decomposer_demo.estate import Estate, expected, grade as check
    a, b = Estate(), Estate()
    ids = expected('estate-moscow-work')
    for id in ids:
        a.fetch(id)
    a.submit_answer('\n'.join(a.rows[id]['url'] for id in ids))
    assert check('estate-moscow-work', a.state())['score'] == 1
    assert b.state()['answer'] is None
    assert check('estate-no-matches', b.state())['score'] == 0
    b.search('moscow', 4, 2000000, min_price=1000000)
    b.submit_answer('No matching listings were found.')
    assert check('estate-no-matches', b.state())['score'] == 1


def test_decomposer_in_process_new_run_wait(monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from gyms.decomposer_demo.run import episode
    from gyms import local_workers as local
    class Model(FakeMessagesListChatModel):
        def bind_tools(self,tools,**kwargs):return self
    ids=iter(['worker-0','run-0'])
    monkeypatch.setattr(local,'uuid4',lambda:next(ids))
    def call(name,args,id):return AIMessage(content='',tool_calls=[{'name':name,'args':args,'id':id,'type':'tool_call'}])
    policy=Model(responses=[call('new',{'agent_type_id':'qwen-4b-calendar'},'new'),
                           call('run',{'agent_id':'worker-0','prompt':'Keep AIRI speakers only'},'run'),
                           call('wait',{},'wait'),AIMessage(content='Done')])
    deletes=[{'name':'remove_event','args':{'id':e['id']},'id':e['id'],'type':'tool_call'} for e in load_task()['calendar']['events'] if e['id'] not in KEEP_IDS]
    worker=Model(responses=[AIMessage(content='',tool_calls=deletes),AIMessage(content='Only AIRI speakers remain')])
    result=asyncio.run(episode(TASK_ID,policy,worker_policy=worker))
    assert result['harness']=='decomposer'
    assert result['status']=='finished',result['error']
    assert result['passed']
    assert result['agent_runs']['run-0']['response']=='Only AIRI speakers remain', (result['agent_runs']['run-0'].get('error'), result['worker_runs'])


def test_local_worker_fork_and_cancellation():
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from gyms.decomposer_demo.local import LocalWorkers
    class Model(FakeMessagesListChatModel):
        def bind_tools(self,tools,**kwargs):return self
    async def run():
        workers=LocalWorkers(Model(responses=[AIMessage(content='Ready')]),[],'Worker')
        t=await workers.create_thread()
        r=await workers.create_run(t['thread_id'],'worker',{'messages':[{'role':'user','content':'Hello'}]})
        await workers.tasks[r['run_id']]
        copy=await workers.copy_thread(t['thread_id'])
        assert copy!=t
        assert workers.states[copy['thread_id']]==workers.states[t['thread_id']]
        workers.states[copy['thread_id']]['messages'].clear()
        assert workers.states[t['thread_id']]['messages']
        await workers.close()
        assert all(t.done() for t in workers.tasks.values())
    asyncio.run(run())


def test_frontend_checker_rejects_invalid_and_unrequested_css():
    from gyms.decomposer_demo.frontend import seed
    from gyms.decomposer_demo.tasks import task_spec

    task_id = 'frontend-roomy'
    task = task_spec(task_id)
    state = seed()
    correct_css = state['/styles.css'] + '\n' + '\n'.join(
        f'{selector} {{ {key}: {value}; }}' for (selector, key), value in task['targets'].items())
    state['/styles.css'] = correct_css
    assert grade(task_id, state)['score'] == 1
    state['/styles.css'] += '\nbody { color: red; }'
    assert grade(task_id, state)['score'] == 0
    state['/styles.css'] = correct_css.replace('max-width: 1100px;', 'max-width: 1100px !important;')
    assert grade(task_id, state)['score'] == 0
    state['/styles.css'] = correct_css + '\n@import "https://example.invalid/x.css";'
    assert grade(task_id, state)['score'] == 0


def test_worker_recovers_from_bad_clock_and_preserves_failed_arguments():
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from gyms.decomposer_demo.local import LocalWorkers
    from gyms.decomposer_demo.run import environment

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    def call(name, args, id):
        return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': id, 'type': 'tool_call'}])

    task, gym, tools, final_state, system = environment(TASK_ID, 'en')
    invalid = {'from': '2026-09-19T12:00:00+03:00', 'to': '2026-09-19T18:00:00+03:00'}
    deletes = [{'name': 'remove_event', 'args': {'id': e['id']}, 'id': e['id'], 'type': 'tool_call'}
               for e in gym.events if e['id'] not in KEEP_IDS]
    model = Model(responses=[call('get_events', invalid, 'bad-read'),
                            call('get_events', {'from': '12:00', 'to': '18:00'}, 'good-read'),
                            AIMessage(content='', tool_calls=deletes), AIMessage(content='Done')])

    async def run():
        workers = LocalWorkers(model, tools, system)
        thread = await workers.create_thread()
        result = await workers.create_run(thread['thread_id'], 'calendar',
                                          {'messages': [{'role': 'user', 'content': task['prompt']}]})
        await workers.tasks[result['run_id']]
        record = await workers.get_run(thread['thread_id'], result['run_id'])
        assert record['status'] == 'success'
        errors = [m for m in record['after'] if m['type'] == 'tool' and m.get('status') == 'error']
        assert len(errors) == 1
        assert 'HH:mm' in errors[0]['content']
        await workers.close()

    asyncio.run(run())
    assert gym.actions[0]['args'] == invalid
    assert gym.actions[0]['error_type'] == 'ValueError'
    assert grade(TASK_ID, final_state())['score'] == 1


def test_worker_fatal_error_keeps_partial_message_history():
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import StructuredTool
    from gyms.decomposer_demo.local import LocalWorkers

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    async def broken(value: str) -> str:
        """Simulate a non-recoverable infrastructure failure."""
        raise RuntimeError('inference infrastructure unavailable')

    model = Model(responses=[AIMessage(content='', tool_calls=[
        {'name': 'broken', 'args': {'value': 'preserve-me'}, 'id': 'failed', 'type': 'tool_call'}])])

    async def run():
        workers = LocalWorkers(model, [StructuredTool.from_function(coroutine=broken)], 'Worker')
        thread = await workers.create_thread()
        result = await workers.create_run(thread['thread_id'], 'worker',
                                          {'messages': [{'role': 'user', 'content': 'Try the tool'}]})
        await workers.tasks[result['run_id']]
        record = await workers.get_run(thread['thread_id'], result['run_id'])
        assert record['status'] == 'error'
        assert record['after'][-1]['tool_calls'][0]['args'] == {'value': 'preserve-me'}
        assert 'RuntimeError' in workers.states[thread['thread_id']]['error']
        await workers.close()

    asyncio.run(run())


@pytest.mark.parametrize('failure, expected_type', [(TimeoutError, 'TimeoutError'), (RuntimeError, 'WorkerError')])
def test_worker_failure_is_visible_without_masking_infrastructure(monkeypatch, failure, expected_type):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from gyms.decomposer_demo.run import episode
    from gyms import local_workers as local

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    class FailingModel(Model):
        async def _agenerate(self, *args, **kwargs):
            raise failure('worker failed')

    ids = iter(['worker-0', 'run-0'])
    monkeypatch.setattr(local, 'uuid4', lambda: next(ids))
    def call(name, args, id):
        return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': id, 'type': 'tool_call'}])
    policy = Model(responses=[call('new', {'agent_type_id': 'qwen-4b-calendar'}, 'new'),
                             call('run', {'agent_id': 'worker-0', 'prompt': 'Keep AIRI speakers'}, 'run'),
                             call('wait', {}, 'wait'), AIMessage(content='Could not finish')])
    result = asyncio.run(episode(TASK_ID, policy, worker_policy=FailingModel(responses=[])))
    assert result['status'] == 'error'
    assert result['error_type'] == expected_type
    assert result['worker_errors'] == [f'{failure.__name__}: worker failed']
    assert not result['passed']
