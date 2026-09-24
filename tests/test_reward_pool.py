from rl.toolathlon_gym.prepare_reward_pool import build_pool


def test_pool_boundaries_and_infrastructure():
    manifest = {"config": {"repetitions": 5}, "tasks": ["never", "sometimes", "always", "infra"], "episodes": {}}
    for task, scores in {"never": [.89]*5, "sometimes": [.9, 0, 0, 0, 0], "always": [1]*5, "infra": [1]*5}.items():
        for i, score in enumerate(scores, 1):
            manifest["episodes"][f"{task}/rep-{i:03d}"] = {
                "status": "infrastructure_error" if task == "infra" and i == 5 else "completed", "partial_score": score}
    rows = build_pool(manifest)["tasks"]
    assert {r["task_id"]: r["successes_ge90"] for r in rows} == {"infra": 4, "sometimes": 1}
