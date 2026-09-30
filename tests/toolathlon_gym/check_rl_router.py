"""Check actual RL AgentLoop routing inside a CPU-only Ray worker; no training."""
import json
import os

import ray


@ray.remote
def check():
    import asyncio
    import tempfile
    from unittest.mock import patch
    from rl.toolathlon_gym.agent_loop import ToolathlonAgentLoop

    class ProbeDone(Exception):
        pass

    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"RL_ARTIFACTS": directory}):
        with patch("rl.toolathlon_gym.agent_loop.Episode") as episode:
            episode.return_value.start.side_effect = ProbeDone
            try:
                asyncio.run(ToolathlonAgentLoop.run(None, {}, extra_info={"task_id": "probe"}))
            except ProbeDone:
                pass
            from decomposer.models import create_model
            model = create_model("qwen_3_5_4b_unlooped_non_thinking")
            assert model.model_name == "Qwen/Qwen3.5-4B-unlooped"
            assert os.environ.get("LLM_PROXY_MASTER_KEY"), "Credential missing in Ray worker"
            model.http_client.close()
            asyncio.run(model.http_async_client.aclose())
            return {"url": model.openai_api_base, "model": model.model_name,
                    "socket_configured": bool(os.environ.get("LLM_PROXY_UNIX_SOCKET")),
                    "credential_present": True}


if __name__ == "__main__":
    ray.init(address="local", num_cpus=1, num_gpus=0, include_dashboard=False)
    try:
        print(json.dumps(ray.get(check.remote()), indent=2))
    finally:
        ray.shutdown()
