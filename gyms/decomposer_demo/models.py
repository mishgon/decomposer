"""Hosted worker assignments from the demo's selectWorkerModel function."""
WORKER_MODELS = {
    'calendar': 'lmrouter/qwen_3_5_4b_unlooped_non_thinking',
    'estate': 'lmrouter/qwen_3_5_4b_unlooped_non_thinking',
    'frontend': 'lmrouter/qwen_3_8_flash_next_non_thinking',
}


def create_hosted_model(name):
    from decomposer.models import create_model
    model = create_model(name)
    model.max_tokens = 4096
    return model
