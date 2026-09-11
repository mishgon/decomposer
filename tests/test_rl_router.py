import os
import subprocess
import unittest


class Router(unittest.TestCase):
    def test_hosted_defaults_and_url_normalization(self):
        env = {"PATH": os.environ["PATH"], "LMROUTER_ENV": "/nonexistent",
               "LLM_PROXY_URL": "https://router/", "LLM_PROXY_MASTER_KEY": "test-key"}
        result = subprocess.run(["bash", "-eu", "-c",
            'source training/toolathlon_gym/router.sh; '
            'printf "%s\\n" "$SUBAGENT_URL" "$SUBAGENT_MODEL"; '
            'test "$VLLM_API_KEY" = test-key'], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["https://router/v1", "Qwen/Qwen3.5-4B"])

    def test_missing_router_fails(self):
        result = subprocess.run(["bash", "-eu", "-c", "source training/toolathlon_gym/router.sh"],
            env={"PATH": os.environ["PATH"], "LMROUTER_ENV": "/nonexistent"}, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
