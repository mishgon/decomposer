import pytest

from opd.wideseek.reward import task_reward


@pytest.mark.parametrize('score,expected', [(0, 0), (.75, 0), (.9, 0), (.900001, 1), (1, 1)])
def test_width_threshold_is_strict_and_preserves_raw_f1(score, expected):
    result = {'status': 'finished', 'evaluation': {'score': score}}
    assert task_reward(result, 'width') == expected
    assert result['evaluation']['score'] == score


@pytest.mark.parametrize('source', ['width', 'depth', 'deepdive-qa_rl'])
def test_timeout_zero_but_missing_judge_score_stays_invalid(source):
    assert task_reward({'status': 'timeout', 'evaluation': {'score': None}}, source) == 0
    with pytest.raises(ValueError, match='Invalid validation score'):
        task_reward({'status': 'error', 'evaluation': {'score': None}}, source)


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -1, 1.1])
def test_bad_scores_are_not_binary_failures(score):
    with pytest.raises(ValueError):
        task_reward({'status': 'finished', 'evaluation': {'score': score}}, 'width')


def test_width_requires_finished_episode():
    assert task_reward({'status': 'timeout', 'evaluation': {'score': 1}}, 'width') == 0


@pytest.mark.parametrize('source', ['depth', 'deepdive-qa_rl', 'deepdive-qa_sft'])
@pytest.mark.parametrize('score', [0, 1])
def test_qa_rewards_unchanged(source, score):
    assert task_reward({'status': 'finished', 'evaluation': {'score': score}}, source) == score
