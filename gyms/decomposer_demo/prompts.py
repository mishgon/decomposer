"""Pinned demo prompt text, with explicit Python-only capability differences."""
import json
from pathlib import Path

SNAPSHOT = json.loads(Path(__file__).with_name('demo_prompts.json').read_text())

CAPABILITIES = {
    'calendar': '',
    'estate': '\n\n## Python benchmark capabilities\nThe archived listings are deterministic fictional records, not live properties. Photo inspection is unavailable and returns no visual evidence. Use the recorded listing fields for this task and disclose the missing visual verification. Follow the user\'s explicit ranking and coverage requirements.',
    'frontend': '\n\n## Python benchmark capabilities\nThe shared project files match the demo seed, but this environment stores source in memory and does not execute it. There is no live preview, build, browser or screenshot. Preview tools report unavailable. Verify requested source changes by reading files; report that visual and functional checks were unavailable.',
}


def presentation(language):
    if language not in {'en', 'ru'}:
        raise ValueError('language must be en or ru')
    return SNAPSHOT['presentation'].replace('in Russian.', 'in English.' if language == 'en' else 'in Russian.')


def environment_prompt(environment, role, language):
    return SNAPSHOT['environments'][environment][role] + CAPABILITIES[environment] + presentation(language)


def coordinator_prompt(language):
    from decomposer.prompts import DECOMPOSER_SYSTEM_PROMPT, RUN_BUDGET_CONVENTION
    core = DECOMPOSER_SYSTEM_PROMPT.replace('{run_budget_convention}',
        RUN_BUDGET_CONVENTION.format(agent_run_budget_seconds=60, agent_shutdown_grace_seconds=60))
    workspace = '\n\n'.join(SNAPSHOT['environments'][name]['workspace'] + CAPABILITIES[name]
                              for name in ('calendar', 'estate', 'frontend'))
    return core + '\n\n# Workspace context\n\n' + workspace + presentation(language)
