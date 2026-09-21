import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from gyms.wideseek.runtime import BudgetExceeded, init_budget, reserve, refund
from gyms.wideseek.evaluate import validate_judge, evaluate
from gyms.wideseek.run import usage
from gyms.wideseek.vendor.table_reward import extract_final_answer, evaluate_markdown


class BudgetTests(unittest.TestCase):
    def test_shared_budget_cannot_be_overspent(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "episode"
            init_budget(path, calls=4, tokens=9000)
            def take(_):
                try: return reserve(path)
                except BudgetExceeded: return 0
            with ThreadPoolExecutor(max_workers=8) as pool:
                spent = list(pool.map(take, range(10)))
            self.assertEqual(sum(spent), 9000)
            refund(path, 100)
            self.assertEqual(reserve(path), 100)
            with self.assertRaises(BudgetExceeded): reserve(path)


class ScoreTests(unittest.IsolatedAsyncioTestCase):
    def test_vendor_functions_match_pinned_upstream(self):
        import ast
        upstream = Path("external/RLinf/rlinf/agents/wideseek_r1/utils/reward.py")
        if not upstream.exists():
            self.skipTest("Optional upstream source checkout is not present")
        def functions(path):
            return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        reference = functions(upstream)
        for name, body in functions(Path("gyms/wideseek/vendor/table_reward.py")).items():
            self.assertEqual(body, reference[name], name)

    async def test_judge_failure_is_unscored_not_zero(self):
        class BrokenJudge:
            http_async_client = None
            async def ainvoke(self, *args, **kwargs):
                raise TimeoutError("test timeout")
            async def aclose(self):
                pass
        judge = BrokenJudge()
        judge.http_async_client = judge
        table = "```markdown\n| Name | Value |\n|---|---|\n| A | yes |\n```"
        with tempfile.TemporaryDirectory() as root, patch("gyms.wideseek.evaluate.model", return_value=judge):
            result = await evaluate({"answer": table, "unique_columns": ["Name"]}, table, Path(root))
            self.assertEqual(result["status"], "judge_error")
            self.assertIsNone(result["item_f1"])
            self.assertGreater(usage(Path(root))["judge"]["errors"], 0)

    async def test_native_perfect_partial_wrong_malformed(self):
        import ast, json
        async def judge(messages):
            text = messages[1]["content"]
            if messages[0]["content"].startswith("You are an expert"):
                pairs = ast.literal_eval(text[text.index("{"):].strip())
                result = {k: int(v["response"] == v["target"]) for k, v in pairs.items()}
            else:
                result = {}  # exact primary keys need no remapping
            return "```json\n" + json.dumps(result) + "\n```"
        def table(rows): return "```markdown\n| Name | Value |\n|---|---|\n" + rows + "\n```"
        reference = {"answer": table("| A | yes |\n| B | no |"), "unique_columns": ["Name"]}
        for rows, expected in [("| A | yes |\n| B | no |", 1.), ("| A | yes |", 2/3), ("| Z | wrong |", 0.)]:
            parsed = extract_final_answer(table(rows), mode="markdown")
            score, ok = await evaluate_markdown(parsed, reference, judge)
            self.assertTrue(ok)
            self.assertAlmostEqual(score, expected)
        self.assertIsNone(extract_final_answer("Not a table", mode="markdown"))

    async def test_bad_judge_output_is_rejected(self):
        messages = [{"content": "You are an expert in grading"},
                    {"content": "Grade: {'idx_0': {'response': 'A', 'target': 'A'}}"}]
        for text in ("Oops", "```json\n{}\n```", '```json\n{"idx_0": 0.5}\n```'):
            with self.assertRaises(ValueError): validate_judge(text, messages)


if __name__ == "__main__": unittest.main()
