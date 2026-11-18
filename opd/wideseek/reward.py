"""OPD reward semantics; preserve the Gym's raw score for auditing."""
import math


WIDTH_REWARD = 'f1>0.9'


def native_score(result):
    score = result.get('evaluation', {}).get('score')
    if score is None and result.get('status') == 'timeout':
        return 0.0
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid validation score; inspect the saved episode result')
    return float(score)


def task_reward(result, source):
    score = native_score(result)
    if source == 'width':
        return float(result.get('status') == 'finished' and score > .9)
    return score
