"""Validation reward entry point; final state must come from the rollout harness."""
from gyms.decomposer_demo import grade


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """veRL-compatible signature; model-written answers never determine the reward."""
    if data_source not in {'decomposer-demo/validation', 'decomposer-demo/en', 'decomposer-demo/ru'}:
        raise ValueError('decomposer-demo tasks are validation-only')
    if not extra_info or 'final_state' not in extra_info:
        raise ValueError('The rollout harness must provide extra_info.final_state')
    language = extra_info.get('language', data_source.rsplit('/', 1)[-1] if data_source != 'decomposer-demo/validation' else 'ru')
    if data_source != 'decomposer-demo/validation' and data_source != f'decomposer-demo/{language}':
        raise ValueError('Validation source and language disagree')
    return grade(ground_truth, extra_info['final_state'], language)
