import json
from pathlib import Path


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trainable_groups(rows: list[dict]) -> list[dict]:
    """Do not update on groups whose advantages are all zero."""
    result = []
    for row in rows:
        rewards = row["rewards"]
        if len(rewards) != len(row["completions"]):
            raise ValueError("Each completion must have exactly one reward")
        if len(rewards) > 1 and min(rewards) != max(rewards):
            result.append(row)
    return result
