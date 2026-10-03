"""Normalize Hub datasets to the numeric-answer environment and freeze the split."""

import argparse
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

from halo_demo.config import Config
from halo_demo.environment import Task, problem_split
from halo_demo.io import read_jsonl, write_json, write_jsonl


def normalize_row(row: dict, config: Config) -> Task:
    prompt = row.get(config.prompt_field)
    answer = row.get(config.answer_field)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{config.prompt_field!r} must contain a nonempty prompt string")
    if isinstance(answer, bool) or not isinstance(answer, (str, int)):
        raise ValueError(f"{config.answer_field!r} must contain an integer or an integer-answer string")
    answer = str(answer).strip()
    if config.answer_delimiter:
        if config.answer_delimiter not in answer:
            raise ValueError(f"Answer is missing delimiter {config.answer_delimiter!r}")
        answer = answer.rsplit(config.answer_delimiter, 1)[1].strip()
    # Accept ordinary integers and correctly grouped thousands, e.g. GSM8K's '#### 1,234'.
    if not re.fullmatch(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)", answer):
        raise ValueError(f"Answer {answer[:80]!r} is not an integer; configure answer_delimiter or use another grader")
    canonical = answer.replace(",", "").lstrip("+-").lstrip("0") or "0"
    if len(canonical) > 9:
        raise ValueError("Answers must fit the grader's nine-digit integer format")
    return Task(prompt.strip(), int(answer.replace(",", "")))


def select_tasks(rows, config: Config, limit: int, excluded: set[str] | None = None) -> list[Task]:
    seen = set(excluded or ())
    tasks = []
    for index, row in enumerate(rows):
        try:
            task = normalize_row(row, config)
        except ValueError as error:
            raise ValueError(f"Dataset row {index}: {error}") from error
        if task.prompt in seen:
            continue
        seen.add(task.prompt)
        tasks.append(task)
        if len(tasks) == limit:
            break
    return tasks


def prepare(config: Config, run_dir: Path):
    if not config.dataset_name:
        train, evaluation = problem_split(config.max_operand, config.eval_size, config.seed)
        train = [Task(p.prompt, p.answer) for p in train]
        evaluation = [Task(p.prompt, p.answer) for p in evaluation]
        metadata = {"source": "synthetic", "seed": config.seed}
    else:
        from datasets import load_dataset

        kwargs = {"path": config.dataset_name, "name": config.dataset_config, "revision": config.dataset_revision}
        training_data = load_dataset(**kwargs, split=config.dataset_train_split)
        metadata = {
            "source": "huggingface",
            **kwargs,
            "train_split": config.dataset_train_split,
            "eval_split": config.dataset_eval_split,
            "train_fingerprint": training_data._fingerprint,
        }
        shuffled = training_data.shuffle(seed=config.seed)
        if config.dataset_eval_split:
            evaluation_data = load_dataset(**kwargs, split=config.dataset_eval_split)
            metadata["eval_fingerprint"] = evaluation_data._fingerprint
            evaluation = select_tasks(evaluation_data.shuffle(seed=config.seed), config, config.eval_size)
            train = select_tasks(shuffled, config, config.dataset_train_limit, {t.prompt for t in evaluation})
        else:
            tasks = select_tasks(shuffled, config, config.eval_size + config.dataset_train_limit)
            evaluation, train = tasks[: config.eval_size], tasks[config.eval_size :]
    if len(evaluation) != config.eval_size or len(train) < config.prompts_per_round:
        raise ValueError(
            "Not enough unique dataset rows after splitting/deduplication; reduce eval_size or prompts_per_round"
        )
    data_dir = run_dir / "data"
    data_dir.mkdir()
    write_jsonl(data_dir / "train.jsonl", [asdict(task) for task in train])
    write_jsonl(data_dir / "eval.jsonl", [asdict(task) for task in evaluation])
    write_json(data_dir / "source.json", {**metadata, "train_count": len(train), "eval_count": len(evaluation)})
    print(f"Prepared {len(train)} training tasks and {len(evaluation)} held-out evaluation tasks.", flush=True)


def read_tasks(run_dir: Path, split: str) -> list[Task]:
    return [Task(**row) for row in read_jsonl(run_dir / "data" / f"{split}.jsonl")]


def resolve_revisions(config: Config) -> Config:
    """Resolve moving Hub refs once so separate phases cannot load different weights/data."""
    from huggingface_hub import HfApi

    api = HfApi()
    changes = {}
    if not re.fullmatch(r"[a-fA-F0-9]{40}", config.model_revision):
        changes["model_revision"] = api.model_info(config.model_name, revision=config.model_revision).sha
    if config.dataset_name and not re.fullmatch(r"[a-fA-F0-9]{40}", config.dataset_revision):
        changes["dataset_revision"] = api.dataset_info(config.dataset_name, revision=config.dataset_revision).sha
    return replace(config, **changes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.run_dir / "config.json"
    config = resolve_revisions(Config(**json.loads(config_path.read_text())))
    write_json(config_path, asdict(config))
    prepare(config, args.run_dir)
