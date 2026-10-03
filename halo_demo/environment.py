"""A dependency-free, one-step RL environment with a verifiable reward."""

import random
import re
from dataclasses import dataclass

SYSTEM_PROMPT = (
    "Solve the arithmetic problem. Reply with exactly <answer>INTEGER</answer>. "
    "For example, for 2 + 3 reply <answer>5</answer>. Do not include explanations."
)
ANSWER_PATTERN = re.compile(r"<answer>(-?(?:0|[1-9][0-9]{0,8}))</answer>")


@dataclass(frozen=True)
class Problem:
    left: int
    operator: str
    right: int

    def __post_init__(self):
        if self.operator not in ("+", "-", "*"):
            raise ValueError("operator must be +, -, or *")

    @property
    def answer(self) -> int:
        if self.operator == "+":
            return self.left + self.right
        if self.operator == "-":
            return self.left - self.right
        return self.left * self.right

    @property
    def prompt(self) -> str:
        return f"What is {self.left} {self.operator} {self.right}?"

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.prompt},
        ]


@dataclass(frozen=True)
class Task:
    """Normalized task from either synthetic arithmetic or a Hugging Face dataset."""

    prompt: str
    answer: int

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.prompt},
        ]


def score(response: str, expected: int, *, truncated: bool = False) -> dict:
    """A capped, full-string parser prevents extra answers or prose from gaming reward."""
    match = ANSWER_PATTERN.fullmatch(response.strip()) if len(response) <= 256 else None
    formatted = match is not None and not truncated
    correct = formatted and int(match[1]) == expected
    return {
        "reward": 1.0 if correct else 0.1 if formatted else 0.0,
        "correct": correct,
        "formatted": formatted,
        "truncated": truncated,
    }


class ArithmeticEnv:
    """reset(problem) -> observation; step(text) -> observation, reward, done, info."""

    def __init__(self):
        self.problem = None
        self.done = True

    def reset(self, problem: Problem | Task) -> list[dict[str, str]]:
        self.problem = problem
        self.done = False
        return problem.messages

    def step(self, response: str, *, truncated: bool = False) -> tuple[str, float, bool, dict]:
        if self.done:
            raise RuntimeError("Call reset() before each one-step episode.")
        result = score(response, self.problem.answer, truncated=truncated)
        self.done = True
        return "Episode finished.", result["reward"], True, result


def problem_split(max_operand: int, eval_size: int, seed: int) -> tuple[list[Problem], list[Problem]]:
    """Partition the finite task space so no evaluation prompt enters training."""
    problems = [
        Problem(a, op, b) for a in range(max_operand + 1) for b in range(max_operand + 1) for op in ("+", "-", "*")
    ]
    if not 0 < eval_size < len(problems):
        raise ValueError("eval_size must leave at least one training problem")
    random.Random(seed).shuffle(problems)
    return problems[eval_size:], problems[:eval_size]
