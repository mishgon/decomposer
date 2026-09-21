import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, AsyncMock, MagicMock
from types import SimpleNamespace

from gyms.wideseek.runtime import BudgetExceeded, init_budget, reserve, refund
from gyms.wideseek.evaluate import validate_judge, evaluate
from gyms.wideseek.run import usage, episode
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
    async def test_rescore_preserves_original_and_selects_judge(self):
        import hashlib, json
        from gyms.wideseek.rescore import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "tasks.jsonl"
            data.write_text(json.dumps({"task_id": "t", "answer": "reference"}) + "\n")
            run = root / "run"
            path = run / "simple/t/attempt-001/result.json"
            path.parent.mkdir(parents=True)
            original = {"mode": "simple", "task_id": "t", "attempt": 1, "status": "finished",
                        "answer": "prediction", "evaluation": {"score": None}}
            path.write_text(json.dumps(original))
            before = path.read_bytes()
            (run / "manifest.json").write_text(json.dumps({"settings": {
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}}))
            judge = AsyncMock(return_value={"score": .5, "status": "scored"})
            args = SimpleNamespace(data=data, run=run, output=root/"new-scores", judge_model="large-judge", concurrency=2)
            with patch("gyms.wideseek.rescore.evaluate", new=judge), \
                    patch("gyms.wideseek.rescore.subprocess.check_output", return_value="test-revision"):
                await main(args)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(judge.await_args.kwargs["judge_model_id"], "large-judge")
            saved = json.loads((args.output / "simple/t/attempt-001/result.json").read_text())
            self.assertEqual(saved["previous_evaluation"], original["evaluation"])
            self.assertEqual(saved["evaluation"]["score"], .5)

    async def test_cancelled_execution_is_not_reused_on_retry(self):
        graph = MagicMock()
        graph.ainvoke = AsyncMock(side_effect=[asyncio.CancelledError(), {}])
        graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={}))
        policy = MagicMock()
        policy.http_async_client.aclose = AsyncMock()
        args = SimpleNamespace(model_calls=2, output_tokens=100, worker_url="http://unused", timeout=1)
        task = {"task_id": "test", "question": "Question", "answer": "Answer", "unique_columns": []}
        with tempfile.TemporaryDirectory() as folder, patch("gyms.wideseek.run.model", return_value=policy), \
                patch("langchain.agents.create_agent", return_value=graph), \
                patch("gyms.wideseek.run.evaluate", new=AsyncMock(return_value={"score": 0.})):
            root = Path(folder)
            with self.assertRaises(asyncio.CancelledError):
                await episode(task, "simple", 1, root, args)
            parent = root / "simple/test/attempt-001"
            old = next(parent.glob("execution-*"))
            self.assertFalse((parent / "result.json").exists())
            await episode(task, "simple", 1, root, args)
            self.assertTrue((old / "trace.json").exists())
            self.assertEqual(len(list(parent.glob("execution-*"))), 2)
            import json
            result = json.loads((parent / "result.json").read_text())
            self.assertNotEqual(result["execution_directory"], old.name)

    async def test_invalid_tool_arguments_are_feedback_not_episode_crash(self):
        from langchain_core.messages import AIMessage
        from langgraph.prebuilt import ToolNode
        from langgraph.graph import StateGraph, MessagesState, START, END
        from gyms.wideseek.worker import access, search
        builder = StateGraph(MessagesState)
        builder.add_node("tools", ToolNode([access, search]))
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        graph = builder.compile()
        for name, arguments in [("access", {"url": "unused", "access_token": 50000}),
                                ("search", {"query": "unused", "topk": 0})]:
            result = await graph.ainvoke({"messages": [
                AIMessage(content="", tool_calls=[{"id": "invalid", "name": name, "args": arguments}])]})
            message = result["messages"][-1]
            self.assertEqual(message.status, "error")
            self.assertIn("Error", message.content)

    async def test_qa_native_score_and_bad_judge(self):
        class Judge:
            def __init__(self, text):
                self.content = text
                self.http_async_client = self
            async def ainvoke(self, *args, **kwargs):
                return self
            def model_dump(self, **kwargs):
                return {"content": self.content}
            async def aclose(self):
                pass
        task = {"question": "Nationality?", "answer": "Australian", "unique_columns": []}
        for reply, status, score in [("Correct", "scored", 1.), ("Incorrect", "scored", 0.),
                                      ("Sorry", "judge_error", None)]:
            with tempfile.TemporaryDirectory() as root, patch("gyms.wideseek.evaluate.model", return_value=Judge(reply)):
                result = await evaluate(task, r"\boxed{Australian}", Path(root))
                self.assertEqual((result["status"], result["score"]), (status, score))
                self.assertEqual(result["metric"], "qa_accuracy")
        with tempfile.TemporaryDirectory() as root:
            result = await evaluate(task, "Australian", Path(root))
            self.assertEqual(result["score"], 0.)
            self.assertFalse(result["format_ok"])

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
            self.assertIsNone(result["score"])
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
