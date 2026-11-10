import pytest

from gyms.decomposer_demo.estate import Estate, expected, grade, snapshot
from gyms.decomposer_demo.tasks import ESTATE, task_spec


def answer_for(task_id, gym):
    ids = expected(task_id)
    for id in ids:
        gym.fetch(id)
    return '\n'.join(f'{i}. [Listing {id}]({gym.rows[id]["url"]})' for i, id in enumerate(ids, 1))


def test_demo_search_contract_and_fictional_records():
    gym = Estate()
    result = gym.search('Москва', 3, 150000, deal_type='rent', sort='price_ascending', min_price=100000)
    assert result['status'] == 'ok'
    assert all(r['rooms'] == 3 and r['price_period'] == 'month' for r in result['results'])
    assert all(r['id'].isdigit() and len(r['id']) <= 12 for r in snapshot())
    assert all(r['condition'] == 'finished and habitable' for r in snapshot())
    assert result['has_more'] is False
    assert result['total_pages'] == 1
    assert [r['price'] for r in result['results']] == sorted(r['price'] for r in result['results'])
    with pytest.raises(ValueError):
        gym.search('Москва', 5, 100000)
    with pytest.raises(ValueError):
        gym.search('Atlantis', 1, 100000)


def test_demo_search_twenty_item_pagination_and_diverse_order():
    gym = Estate()
    base = next(iter(gym.rows.values()))
    gym.rows = {str(1000+i): {**base, 'id': str(1000+i), 'price': i*1000+1} for i in range(45)}
    first = gym.search('moscow', 1, 50000, deal_type='rent')
    second = gym.search('moscow', 1, 50000, deal_type='rent', page=2)
    third = gym.search('moscow', 1, 50000, deal_type='rent', page=3)
    assert [len(p['results']) for p in (first, second, third)] == [20, 20, 5]
    assert [p['next_page'] for p in (first, second, third)] == [2, 3, None]
    ids = [r['id'] for p in (first, second, third) for r in p['results']]
    assert len(set(ids)) == 45
    assert first == gym.search('moscow', 1, 50000, deal_type='rent')


def test_fetch_matches_demo_model_visible_response_and_photos_are_honest():
    gym = Estate()
    id = snapshot()[0]['id']
    result = gym.fetch(id)
    assert result['status'] == 'ok' and result['coverage'] == 'search_page'
    assert result['text'] and result['url']
    assert 'property' not in result  # Demo keeps display metadata outside model context.
    photo = gym.inspect_photos(id)
    assert photo['status'] == 'vision_unavailable'
    assert photo['inspected_images'] == photo['requested_images'] == 0
    assert gym.fetch('999999999999')['status'] == 'not_archived'
    with pytest.raises(ValueError):
        gym.fetch('mos-rent-00')


def test_final_coordinator_shortlist_is_graded_with_fetched_evidence():
    for entry in ESTATE:
        task_id = entry[0]
        if task_id == 'estate-no-matches':
            continue
        gym = Estate()
        answer = answer_for(task_id, gym)
        assert grade(task_id, gym.state())['score'] == 0
        gym.submit_answer(answer)
        assert grade(task_id, gym.state())['score'] == 1
        state = gym.state()
        state['fetched'] = []
        assert grade(task_id, state)['score'] == 0
        gym.submit_answer(answer + '\nhttps://invented.example.invalid/99')
        assert grade(task_id, gym.state())['score'] == 0
        gym.submit_answer('\n'.join(reversed(answer.splitlines())))
        if len(expected(task_id)) > 1:
            assert grade(task_id, gym.state())['score'] == 0


def test_no_match_requires_search_and_nonempty_final_answer():
    gym = Estate()
    gym.submit_answer('No suitable matches in the snapshot.')
    assert grade('estate-no-matches', gym.state())['score'] == 0
    gym.search('moscow', 4, 2000000, min_price=1000000)
    assert grade('estate-no-matches', gym.state())['score'] == 1
    gym.submit_answer('Done')
    assert grade('estate-no-matches', gym.state())['score'] == 0
    gym.submit_answer('')
    assert grade('estate-no-matches', gym.state())['score'] == 0


def test_prompts_use_natural_constraints_and_no_extra_submission_tool():
    assert not hasattr(Estate, 'save_shortlist')
    for task_id, *_ in ESTATE:
        for language in ('en', 'ru'):
            prompt = task_spec(task_id, language)['prompt']
            assert 'save_shortlist' not in prompt
            assert '{' not in prompt
            assert 'photo' in prompt.lower() or 'фото' in prompt.lower()
