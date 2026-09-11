"""Check actual RL AgentLoop routing inside a CPU-only Ray worker; no training."""
import json
import os

import ray


@ray.remote
def check():
    import asyncio
    import tempfile
    from unittest.mock import patch
    from training.toolathlon_gym.agent_loop import ToolathlonAgentLoop

    class ProbeDone(Exception):
        pass

    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"RL_ARTIFACTS": directory}):
        with patch("training.toolathlon_gym.agent_loop.Episode") as episode:
            episode.return_value.start.side_effect = ProbeDone
            try:
                asyncio.run(ToolathlonAgentLoop.run(None, {}, extra_info={"task_id": "probe"}))
            except ProbeDone:
                pass
            kwargs = episode.call_args.kwargs
            assert kwargs["subagent_url"] == os.environ["SUBAGENT_URL"]
            assert kwargs["subagent_model"] == "Qwen/Qwen3.5-4B"
            assert os.environ.get("VLLM_API_KEY"), "Credential missing in Ray worker"
            return {"url": kwargs["subagent_url"], "model": kwargs["subagent_model"],
                    "host": kwargs["subagent_host"], "credential_present": True}


if __name__ == "__main__":
    ray.init(address="local", num_cpus=1, num_gpus=0, include_dashboard=False)
    try:
        print(json.dumps(ray.get(check.remote()), indent=2))
    finally:
        ray.shutdown()
