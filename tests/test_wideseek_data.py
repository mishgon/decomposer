import unittest
from gyms.wideseek.prepare import agent_input, normalize


class DatasetTests(unittest.TestCase):
    def test_width_and_reference_isolation(self):
        task = normalize({"question": "Find cities", "answer": "secret reference",
                          "unique_columns": ["City"]}, "width", 0)
        self.assertEqual(task["task_id"], "width-00000")
        self.assertEqual(agent_input(task), {"messages": [{"role": "user", "content": "Find cities"}]})

    def test_depth_metadata(self):
        task = normalize({"question": "Who?", "answer": "A", "aug_answer": ["Alias"], "qid": 7}, "depth", 1)
        self.assertEqual(task["metadata"]["aug_answer"], ["Alias"])
        self.assertEqual(task["unique_columns"], [])

    def test_question_hash_groups_duplicates_across_sources(self):
        row = {"question": "Who?", "answer": "A"}
        self.assertEqual(normalize(row, "width", 0)["question_sha256"],
                         normalize(row, "hybrid", 15)["question_sha256"])

    def test_invalid_row_rejected(self):
        with self.assertRaises(ValueError):
            normalize({"question": "Who?"}, "width", 0)


if __name__ == "__main__":
    unittest.main()
