import os
import subprocess
import unittest


class Router(unittest.TestCase):
    def test_exports_only_registry_credentials(self):
        env = {"PATH": os.environ["PATH"], "LMROUTER_ENV": "/nonexistent",
               "LLM_PROXY_URL": "https://router/", "LLM_PROXY_MASTER_KEY": "test-key"}
        result = subprocess.run(["bash", "-eu", "-c",
            'source rl/toolathlon_gym/router.sh; '
            'test "$LLM_PROXY_MASTER_KEY" = test-key; '
            'test -z "${SUBAGENT_URL:-}${SUBAGENT_MODEL:-}${VLLM_API_KEY:-}"'], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_missing_router_fails(self):
        result = subprocess.run(["bash", "-eu", "-c", "source rl/toolathlon_gym/router.sh"],
            env={"PATH": os.environ["PATH"], "LMROUTER_ENV": "/nonexistent"}, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
