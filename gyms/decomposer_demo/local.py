"""Demo-specific worker routing and coordinator configuration."""
import asyncio
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from decomposer import core
from gyms.local_workers import LocalWorkers


class LocalWorkerPool:
    """Route the demo's three agent types to their private episode environments."""
    def __init__(self, workers):
        self.workers = workers

    def get_async(self, agent_type_id):
        return self.workers[agent_type_id]

    @property
    def records(self):
        return {id: record for worker in self.workers.values() for id, record in worker.records.items()}

    async def close(self):
        await asyncio.gather(*(worker.close() for worker in self.workers.values()))


def decomposer(policy, workers, language="ru"):
    from .prompts import SNAPSHOT, coordinator_prompt
    types = [{'agent_type_id': f'qwen-4b-{name}', 'assistant_id': f'qwen-4b-{name}',
              'description': SNAPSHOT['environments'][name]['description']}
             for name in ('calendar', 'estate', 'frontend')]
    middleware = core.DecomposerAgentMiddleware(types, 80, 60, 60)
    # Reuse upstream schemas, duplicate-task guards, review/stopping rules and wait logic.
    middleware.tools = [core._build_new_tool({t['agent_type_id']: t for t in types}, workers),
                        core._build_fork_tool(workers), core._build_run_tool(workers, 80, 60, 60),
                        core._build_wait_tool(workers)]
    prompt = coordinator_prompt(language)
    return create_agent(policy, tools=[], system_prompt=prompt, middleware=[middleware],
                        checkpointer=InMemorySaver())
