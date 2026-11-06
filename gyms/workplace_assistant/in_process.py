"""Native Workplace tables and verifier, without Gym HTTP services."""
import asyncio
from copy import deepcopy
import json
import time
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import message_to_dict
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from decomposer import core
from decomposer.prompts import AGENT_SYSTEM_PROMPT, DECOMPOSER_SYSTEM_PROMPT
from gyms.local_workers import LocalWorkers
from resources_servers.workplace_assistant.utils import get_tools, is_correct

TOOLKITS = ['email', 'calendar', 'analytics', 'project_management', 'customer_relationship_manager']
SYSTEM = ("Today's date is Thursday, 2023-11-30 and the current time is 23:59:00. "
          "Remember the current date and time when answering queries. "
          "Meetings must not start before 9am or end after 6pm.")


class Workplace:
    def __init__(self):
        self.environment = get_tools(TOOLKITS)
        self.actions = []

    def tools(self):
        def wrap(name):
            async def call(**args):
                # Synchronous table mutations on one event loop are atomic across workers.
                self.actions.append({'name': name, 'arguments': json.dumps(deepcopy(args))})
                try:
                    return self.environment['functions'][name](**args)
                except Exception as error:
                    # Match the native resource server's recoverable tool errors.
                    return str(error)
            return call
        return [StructuredTool(name=s['name'], description=s['description'],
                               args_schema=s['parameters'], coroutine=wrap(s['name']))
                for s in self.environment['schemas']]

    def score(self, reference):
        # Use calls actually executed, including runs whose worker history failed.
        return bool(is_correct(self.actions, reference, ''))


async def episode(row, policy, worker_policy, *, timeout=300):
    gym = Workplace()
    workers = LocalWorkers(worker_policy, gym.tools(), AGENT_SYSTEM_PROMPT, max_model_calls=100)
    types = [{'agent_type_id': 'qwen35_4b_non_thinking', 'assistant_id': 'qwen35_4b_non_thinking',
              'description': 'Qwen3.5-4B non-thinking workplace agent with access to the environment tools.'}]
    middleware = core.DecomposerAgentMiddleware(types, 1000, None, 60)
    middleware.tools = [core._build_new_tool({t['agent_type_id']: t for t in types}, workers),
                        core._build_fork_tool(workers), core._build_run_tool(workers, 1000, None, 60),
                        core._build_wait_tool(workers)]
    agent = create_agent(policy, tools=[],
        system_prompt=DECOMPOSER_SYSTEM_PROMPT.replace('{run_budget_convention}\n\n', ''),
        middleware=[middleware, ModelCallLimitMiddleware(run_limit=100, exit_behavior='error')],
        checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': str(uuid4())}, 'recursion_limit': 1000}
    started = time.time()
    error = None
    try:
        await asyncio.wait_for(agent.ainvoke({'messages': [{'role': 'user', 'content':
            SYSTEM + '\n\n' + row['prompt']}]}, config=config), timeout)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        await workers.close()
    worker_errors = [r['error'] for r in workers.records.values() if r.get('error')]
    if worker_errors and error is None:
        error = 'WorkerError: ' + '; '.join(worker_errors)
    state = await agent.aget_state(config)
    native_correct = gym.score(row['ground_truth'])
    return {'task_id': row['id'], 'environment': row['environment_name'],
            'status': 'error' if error else 'finished', 'error': error,
            'error_type': error.split(':', 1)[0] if error else None,
            'score': float(native_correct and error is None),
            'passed': native_correct and error is None, 'native_correct': native_correct,
            'elapsed_seconds': time.time() - started, 'actions': gym.actions,
            'worker_errors': worker_errors,
            'worker_runs': workers.records,
            'messages': [message_to_dict(m) for m in state.values.get('messages', [])]}
