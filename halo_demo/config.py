import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision: str = "7ae557604adf67be50417f59c2c2f167def9a775"
    seed: int = 42
    rounds: int = 5
    prompts_per_round: int = 32
    num_generations: int = 4
    eval_size: int = 64
    max_operand: int = 30
    max_new_tokens: int = 48
    max_prompt_tokens: int = 512
    enable_thinking: bool = False
    learning_rate: float = 5e-6
    gradient_accumulation_steps: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    dataset_name: str | None = None
    dataset_config: str | None = None
    dataset_revision: str = "main"
    dataset_train_split: str = "train"
    dataset_eval_split: str | None = None
    prompt_field: str = "question"
    answer_field: str = "answer"
    answer_delimiter: str | None = None
    dataset_train_limit: int = 2048

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "lora_target_modules":
                if (
                    not isinstance(value, list)
                    or not value
                    or any(not isinstance(item, str) or not item for item in value)
                ):
                    raise ValueError("lora_target_modules must be a nonempty list of module names")
            elif name == "enable_thinking":
                if type(value) is not bool:
                    raise ValueError("enable_thinking must be a boolean")
            elif name in ("dataset_name", "dataset_config", "dataset_eval_split", "answer_delimiter"):
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValueError(f"{name} must be a nonempty string or None")
            elif name in (
                "model_name",
                "model_revision",
                "dataset_revision",
                "dataset_train_split",
                "prompt_field",
                "answer_field",
            ):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be a nonempty string")
            elif name == "learning_rate":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError("learning_rate must be finite and positive")
            elif type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a {'nonnegative' if name == 'seed' else 'positive'} integer")
        if self.max_operand > 100:
            raise ValueError("max_operand must be <= 100 to bound the task pool")
        if self.num_generations < 2:
            raise ValueError("GRPO requires at least two generations per prompt")
        pool_size = 3 * (self.max_operand + 1) ** 2
        if not self.dataset_name and (
            self.eval_size >= pool_size or self.prompts_per_round > pool_size - self.eval_size
        ):
            raise ValueError("The task pool must fit evaluation plus prompts_per_round")
        if self.dataset_name and self.dataset_train_limit < self.prompts_per_round:
            raise ValueError("dataset_train_limit must be at least prompts_per_round")


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as handle:
        return Config(**tomllib.load(handle))
