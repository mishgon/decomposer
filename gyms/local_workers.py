"""Episode-local worker transport shared by in-process gyms."""
import asyncio
from contextvars import ContextVar
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import convert_to_messages
from decomposer.run_budget import RunBudgetContext, RunBudgetMiddleware

worker_id = ContextVar('local_worker_id', default='react')


class LocalWorkers:
    """Implement only the async thread/run operations used by core.py, without HTTP."""
    def __init__(self, policy, tools, system, *, max_model_calls=None):
        self.worker = create_agent(policy, tools=tools, system_prompt=system,
                                  context_schema=RunBudgetContext,
                                  middleware=[RunBudgetMiddleware(), *([ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="error")] if max_model_calls else [])])
        self.states, self.records, self.tasks = {}, {}, {}
        self.threads = SimpleNamespace(create=self.create_thread, copy=self.copy_thread,
            get_state=self.get_state, get_history=self.get_history, get=self.get_thread)
        self.runs = SimpleNamespace(create=self.create_run, get=self.get_run)

    def get_async(self, agent_type_id):
        return self

    async def create_thread(self):
        thread_id = str(uuid4())
        self.states[thread_id] = {'messages': [], 'error': None}
        return {'thread_id': thread_id}

    async def copy_thread(self, thread_id):
        new = await self.create_thread()
        self.states[new['thread_id']] = deepcopy(self.states[thread_id])
        return new

    async def get_state(self, thread_id):
        state = self.states[thread_id]
        return {'checkpoint': {'id': thread_id} if state['messages'] else None,
                'values': deepcopy(state)}

    async def get_thread(self, thread_id):
        return deepcopy(self.states[thread_id])

    async def create_run(self, thread_id, assistant_id, input, config=None, context=None,
                         multitask_strategy='reject'):
        if any(r['thread_id'] == thread_id and r['status'] == 'running' for r in self.records.values()):
            raise ValueError('Worker already has an active run')
        run_id = str(uuid4())
        before = deepcopy(self.states[thread_id]['messages'])
        record = {'run_id': run_id, 'thread_id': thread_id, 'status': 'running', 'before': before}
        self.records[run_id] = record

        async def execute():
            token=worker_id.set(thread_id)
            status = 'running'
            messages = deepcopy(before)
            try:
                async for state in self.worker.astream(
                    {'messages': convert_to_messages(before) + convert_to_messages(input['messages'])},
                    config=config, context=context, stream_mode='values'):
                    messages = [m.model_dump(mode='json') for m in state['messages']]
                self.states[thread_id]['error'] = None
                status = 'success'
            except asyncio.CancelledError:
                status = 'interrupted'
                raise
            except Exception as error:
                self.states[thread_id]['error'] = f'{type(error).__name__}: {error}'
                record['error'] = self.states[thread_id]['error']
                status = 'error'
            finally:
                # Failed calls remain in the run trace; retry history must not
                # contain an assistant tool call without its corresponding result.
                if status == 'success':
                    self.states[thread_id]['messages'] = messages
                record.update(status=status, after=deepcopy(messages))
                worker_id.reset(token)
        self.tasks[run_id] = asyncio.create_task(execute())
        return dict(record)

    async def get_run(self, thread_id, run_id):
        record = self.records[run_id]
        if record['thread_id'] != thread_id:
            raise ValueError('Run does not belong to this worker')
        return dict(record)

    async def get_history(self, thread_id, limit, metadata):
        r = self.records[metadata['run_id']]
        if r['thread_id'] != thread_id:
            raise ValueError('Run does not belong to this worker')
        return [{'metadata': {'source': 'loop'}, 'values': {'messages': deepcopy(r['after'])}},
                {'metadata': {'source': 'input'}, 'values': {'messages': deepcopy(r['before'])}}]

    async def close(self):
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
