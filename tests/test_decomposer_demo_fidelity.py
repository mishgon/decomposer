import json

import pytest

from gyms.decomposer_demo.frontend import Frontend, worker_id
from gyms.decomposer_demo.prompts import SNAPSHOT, coordinator_prompt, environment_prompt


def test_demo_helpers_reach_both_roles_and_languages():
    for environment in ('calendar', 'estate', 'frontend'):
        for language, output_language in (('en', 'English'), ('ru', 'Russian')):
            coordinator = coordinator_prompt(language)
            worker = environment_prompt(environment, 'system', language)
            assert SNAPSHOT['environments'][environment]['workspace'] in coordinator
            assert SNAPSHOT['environments'][environment]['system'] in worker
            assert SNAPSHOT['helpers']['taskApproach'] in coordinator
            assert SNAPSHOT['helpers']['parallelApproach'] in coordinator
            assert SNAPSHOT['helpers']['taskApproach'] in worker
            assert f'answers and reports in {output_language}.' in coordinator
            assert f'answers and reports in {output_language}.' in worker
    assert SNAPSHOT['helpers']['frontendDelegation'] in coordinator_prompt('en')
    assert 'does not execute it' in environment_prompt('frontend', 'system', 'en')
    assert 'returns no visual evidence' in environment_prompt('estate', 'system', 'en')


def test_coordinator_receives_demo_workspace_and_agent_type(monkeypatch):
    from gyms.decomposer_demo import local
    captured = {}
    def create(policy, **kwargs):
        captured.update(kwargs)
        return kwargs
    monkeypatch.setattr(local, 'create_agent', create)
    local.decomposer(object(), object(), 'ru')
    assert captured['system_prompt'] == coordinator_prompt('ru')
    schema = captured['middleware'][0].tools[0].args_schema.model_json_schema()
    assert all(f'qwen-4b-{name}' in json.dumps(schema) for name in ('calendar','estate','frontend'))


def test_frontend_demo_revision_tombstones_and_response_shapes():
    gym = Frontend('en')
    initial = gym.list_files()
    assert initial['version'] == 0 and initial['deleted'] == []
    assert all(f['revision'] == 1 and f['characters'] > 0 for f in initial['files'])
    worker_id.set('owner')
    gym.read_file('/README.md')
    assert gym.delete_file('/README.md') == {'path':'/README.md', 'revision':2, 'deleted':True}
    worker_id.set('other')
    with pytest.raises(ValueError, match='Read'):
        gym.write_file('/README.md', 'replacement')
    with pytest.raises(ValueError, match='Revision conflict'):
        gym.write_file('/README.md', 'replacement', 0)
    assert gym.list_files()['deleted'] == [{'path':'/README.md','revision':2}]
    worker_id.set('owner')
    result = gym.write_file('/README.md', 'replacement')
    assert result == {'path':'/README.md','revision':3,'deleted':False}
    assert gym.list_files()['version'] == 2
    assert gym.get_preview_state()['status'] == 'unavailable'
    with pytest.raises(ValueError, match='unavailable'):
        gym.inspect_screenshot()
    worker_id.set('react')


def test_frontend_limits_and_ambiguous_overlapping_edit_are_atomic():
    gym = Frontend('en')
    for path in ('/./x', '/a/../x', '/a b', '/x/' + 'a'*160):
        with pytest.raises(ValueError): gym.write_file(path, 'x')
    gym.write_file('/new.txt', 'aaa')
    with pytest.raises(ValueError, match='more than once'):
        gym.edit_file('/new.txt', 'aa', 'x')
    with pytest.raises(ValueError, match='100,000'):
        gym.write_file('/new.txt', 'x'*100001)
    assert gym.files['/new.txt'] == 'aaa' and gym.revisions['/new.txt'] == 1


def test_demo_worker_assignments_and_output_cap(monkeypatch):
    from types import SimpleNamespace
    from gyms.decomposer_demo.models import WORKER_MODELS, create_hosted_model
    monkeypatch.setattr('decomposer.models.create_model', lambda name: SimpleNamespace(name=name))
    assert WORKER_MODELS['calendar'] == WORKER_MODELS['estate']
    assert 'flash_next' in WORKER_MODELS['frontend']
    model = create_hosted_model(WORKER_MODELS['frontend'])
    assert model.max_tokens == 4096


def test_estate_schema_exposes_demo_enums_and_ranges():
    from gyms.decomposer_demo.run import environment
    tools = environment('estate-moscow-work', 'en')[2]
    search = next(t for t in tools if t.name == 'search').args_schema.model_json_schema()
    assert set(search['required']) == {'city', 'rooms', 'max_price'}
    assert search['properties']['rooms']['minimum'] == 1
    assert search['properties']['rooms']['maximum'] == 4
    assert search['properties']['deal_type']['enum'] == ['sale', 'rent']
    assert search['properties']['sort']['default'] == 'diverse'
