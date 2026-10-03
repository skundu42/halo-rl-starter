import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from halo_demo.config import Config
from halo_demo.data import normalize_row, prepare, read_tasks, select_tasks


class DataTests(unittest.TestCase):
    def test_hf_column_mapping_and_gsm8k_answer_extraction(self):
        config = replace(Config(), prompt_field="problem", answer_field="solution", answer_delimiter="####")
        task = normalize_row({"problem": "How many?", "solution": "First #### 2. Final #### 1,234"}, config)
        self.assertEqual(task.answer, 1234)
        self.assertEqual(task.prompt, "How many?")
        self.assertNotIn("1234", task.messages[-1]["content"])

    def test_integer_answers(self):
        for answer, expected in [(12, 12), ("-6", -6), (" +42 ", 42), ("0", 0)]:
            self.assertEqual(normalize_row({"question": "q", "answer": answer}, Config()).answer, expected)

    def test_invalid_rows_fail_instead_of_silently_misgrading(self):
        for answer in [None, True, 1.2, "1.2", "1,23", "five", "1000000000", {"value": 5}]:
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                normalize_row({"question": "q", "answer": answer}, Config())
        with self.assertRaisesRegex(ValueError, "delimiter"):
            normalize_row({"question": "q", "answer": "42"}, replace(Config(), answer_delimiter="####"))
        with self.assertRaisesRegex(ValueError, "prompt"):
            normalize_row({"question": [], "answer": 3}, Config())

    def test_deduplicate_and_exclude_eval_prompts(self):
        rows = [{"question": q, "answer": 1} for q in ["held-out", "a", "a", "b", "c"]]
        tasks = select_tasks(rows, Config(), 2, excluded={"held-out"})
        self.assertEqual([t.prompt for t in tasks], ["a", "b"])

    def test_frozen_synthetic_data_is_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            prepare(Config(), path)
            train = read_tasks(path, "train")
            evaluation = read_tasks(path, "eval")
            self.assertEqual(len(evaluation), 64)
            self.assertFalse({t.prompt for t in train} & {t.prompt for t in evaluation})


if __name__ == "__main__":
    unittest.main()
