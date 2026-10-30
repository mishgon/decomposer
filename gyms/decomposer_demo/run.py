"""Python-only Decomposer episodes with episode-local tools and worker state."""
import asyncio
from uuid import uuid4

from . import grade
from .tasks import task_spec


def environment(task_id, language):
    from copy import deepcopy
    from langchain_core.tools import StructuredTool, ToolException
    task=task_spec(task_id,language)
    name=task['environment']
    if name=='calendar':
        from .calendar import Calendar,load_task
        gym=Calendar(load_task(task_id,language))
        read=StructuredTool(name='get_events',description=gym.get_events.__doc__,
            args_schema={'type':'object','properties':{'from':{'type':'string','pattern':r'^(?:[01]\d|2[0-3]):[0-5]\d$|^24:00$', 'description':'Start on the fixed conference day, HH:mm'},
                'to':{'type':'string','pattern':r'^(?:[01]\d|2[0-3]):[0-5]\d$|^24:00$', 'description':'End on the fixed conference day, HH:mm; at most six hours after from'}},
                         'required':['from','to'],'additionalProperties':False},
            func=lambda **args:gym.get_events(args['from'],args['to']))
        tools=[read,*[StructuredTool.from_function(fn) for fn in (gym.remove_event,gym.create_event)]]
        state=lambda:gym.events
    elif name=='estate':
        from .estate import Estate
        gym=Estate();state=gym.state
        tools=[StructuredTool.from_function(fn) for fn in (gym.search,gym.fetch,gym.inspect_photos)]
    else:
        from .frontend import Frontend
        gym=Frontend(language);state=gym.state
        tools=[StructuredTool.from_function(fn) for fn in (gym.list_files,gym.read_file,gym.write_file,gym.edit_file,gym.delete_file,gym.get_preview_state,gym.inspect_screenshot)]
    from .prompts import SNAPSHOT, environment_prompt
    system = environment_prompt(name, 'system', language)
    # These tools only mutate memory. Run each call on the event loop so revision
    # checks and writes cannot interleave across LangChain's executor threads.
    def in_process(name, fn):
        async def invoke(**args):
            try:
                return fn(**args)
            except ValueError as error:
                gym.actions.append({'name': name, 'args': deepcopy(args),
                                    'error': str(error), 'error_type': type(error).__name__})
                raise ToolException(str(error)) from error
        return invoke
    for tool in tools:
        tool.description = SNAPSHOT['environments'][name]['tool_descriptions'].get(tool.name, tool.description)
        if tool.name == 'inspect_photos':
            tool.description += ' This synthetic benchmark has no images; inspection returns unavailable.'
        tool.coroutine = in_process(tool.name, tool.func)
        tool.handle_tool_error = True
    return task,gym,tools,state,system


async def episode(task_id, policy, *, worker_policy=None, language='ru', harness='decomposer',
                  timeout=300, recursion_limit=160):
    from langchain.agents import create_agent
    from langchain_core.messages import message_to_dict
    from langgraph.checkpoint.memory import InMemorySaver
    from .local import LocalWorkers,LocalWorkerPool,decomposer

    task,gym,tools,final_state,system=environment(task_id,language)
    workers=None
    owned_models = {}
    if harness=='react':
        agent=create_agent(policy,tools=tools,system_prompt=system,checkpointer=InMemorySaver())
    elif harness=='decomposer':
        from .tasks import task_ids
        environments = {task['environment']: (task,gym,tools,final_state,system)}
        for candidate in task_ids():
            name = task_spec(candidate, language)['environment']
            if name not in environments:
                environments[name] = environment(candidate, language)
        from .models import WORKER_MODELS, create_hosted_model
        if worker_policy is None:
            owned_models = {name: create_hosted_model(name) for name in set(WORKER_MODELS.values())}
        workers = LocalWorkerPool({f'qwen-4b-{name}': LocalWorkers(worker_policy if worker_policy is not None else owned_models[WORKER_MODELS[name]], values[2], values[4])
                                   for name, values in environments.items()})
        agent=decomposer(policy,workers,language)
    else:
        raise ValueError('harness must be decomposer or react')
    config={'configurable':{'thread_id':str(uuid4())},'recursion_limit':recursion_limit}
    error=None;state={}
    try:
        state=await asyncio.wait_for(agent.ainvoke(
            {'messages':[{'role':'user','content':task['prompt']}]},config=config),timeout)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    finally:
        if workers:
            await workers.close()
        for model in owned_models.values():
            model.http_client.close()
            await model.http_async_client.aclose()
        snapshot=await agent.aget_state(config)
        state=state or dict(snapshot.values)
    worker_errors = [r['error'] for r in workers.records.values() if r['status'] == 'error'] if workers else []
    if worker_errors and error is None:
        types = [message.split(':', 1)[0] for message in worker_errors]
        expected = {'RolloutBudgetExceeded', 'GraphRecursionError', 'TimeoutError'}
        error_type = types[0] if all(kind in expected for kind in types) else 'WorkerError'
        error = error_type + ': worker failures: ' + '; '.join(worker_errors)
    if task['environment'] == 'estate':
        messages = state.get('messages', [])
        answer = next((m.content for m in reversed(messages) if m.type == 'ai' and not m.tool_calls), '')
        gym.submit_answer(answer if isinstance(answer, str) else '')
    from .prompts import SNAPSHOT, coordinator_prompt
    result=grade(task_id,final_state(),language)
    return {'task_id':task_id,'benchmark':'decomposer-demo','environment':task['environment'],
            'language':language,'harness':harness,
            'demo_commit':SNAPSHOT['demo_commit'],
            'system_prompts':{'workers':{name:values[4] for name,values in environments.items()} if workers else {task['environment']:system}, 'coordinator':coordinator_prompt(language) if harness=='decomposer' else None},
            'status':'error' if error else 'finished',
            'error':error,'error_type':error.split(':')[0] if error else None,
            **result,'passed':result['passed'] and error is None,
            'messages':[message_to_dict(m) for m in state.get('messages',[])],
            'agent_runs':state.get('agent_runs',{}), 'worker_runs':workers.records if workers else {},
            'worker_errors':worker_errors, 'actions':gym.actions,'final_state':final_state(),
            'environments':{name:{'actions':values[1].actions,'final_state':values[3]()} for name,values in environments.items()} if workers else {task['environment']:{'actions':gym.actions,'final_state':final_state()}}}
