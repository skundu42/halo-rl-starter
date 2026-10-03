import unittest

from halo_demo.config import Config, load_config
from halo_demo.environment import ArithmeticEnv, Problem, problem_split, score
from halo_demo.io import trainable_groups


class EnvironmentTests(unittest.TestCase):
    def test_arithmetic_and_episode_lifecycle(self):
        env = ArithmeticEnv()
        with self.assertRaises(RuntimeError):
            env.step("<answer>1</answer>")
        for problem, answer in [(Problem(3, "+", 9), 12), (Problem(3, "-", 9), -6), (Problem(3, "*", 9), 27)]:
            observation = env.reset(problem)
            self.assertEqual(observation[-1]["content"], problem.prompt)
            _, reward, done, info = env.step(f"<answer>{answer}</answer>")
            self.assertEqual(reward, 1.0)
            self.assertTrue(done)
            self.assertTrue(info["correct"])
            with self.assertRaises(RuntimeError):
                env.step(f"<answer>{answer}</answer>")

    def test_wrong_valid_answer_only_gets_format_reward(self):
        result = score("<answer>5</answer>", 6)
        self.assertEqual(result["reward"], 0.1)
        self.assertFalse(result["correct"])
        self.assertTrue(result["formatted"])

    def test_malformed_and_reward_hacking_attempts(self):
        for response in [
            "5",
            "<answer>05</answer>",
            "<answer>5.0</answer>",
            "<answer>5e0</answer>",
            "<answer>5</answer><answer>6</answer>",
            "Explanation <answer>5</answer>",
            "<answer>5</answer> more text",
            "<answer>５</answer>",
            "<answer>" + "5" * 10000 + "</answer>",
            "<answer>__import__('os')</answer>",
            "",
        ]:
            with self.subTest(response=response[:80]):
                self.assertEqual(score(response, 5)["reward"], 0.0)
        self.assertEqual(score(" \n<answer>5</answer>\n", 5)["reward"], 1.0)

    def test_truncated_output_cannot_earn_reward(self):
        self.assertEqual(score("<answer>5</answer>", 5, truncated=True)["reward"], 0.0)

    def test_disjoint_deterministic_splits(self):
        train, evaluation = problem_split(10, 20, 42)
        self.assertFalse(set(train) & set(evaluation))
        self.assertEqual(len(train) + len(evaluation), 363)
        self.assertEqual((train, evaluation), problem_split(10, 20, 42))
        self.assertNotEqual(evaluation, problem_split(10, 20, 43)[1])

    def test_no_signal_groups_are_removed(self):
        rows = [
            {"completions": ["a", "b"], "rewards": [0.0, 0.0]},
            {"completions": ["a", "b"], "rewards": [1.0, 1.0]},
            {"completions": ["a", "b"], "rewards": [0.1, 1.0]},
        ]
        self.assertEqual(trainable_groups(rows), [rows[2]])
        with self.assertRaises(ValueError):
            trainable_groups([{"completions": ["a"], "rewards": [1.0, 0.0]}])

    def test_config_validation(self):
        self.assertEqual(load_config("configs/default.toml"), Config())
        for kwargs in [
            {"num_generations": 1},
            {"rounds": 0},
            {"seed": -1},
            {"eval_size": 5000},
            {"learning_rate": float("nan")},
            {"learning_rate": True},
            {"max_operand": 1000},
            {"rounds": True},
            {"lora_rank": 1.5},
            {"model_name": ""},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
