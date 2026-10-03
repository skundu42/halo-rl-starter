"""Lightweight CLI; heavyweight GPU imports live in the worker process."""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from halo_demo.config import load_config
from halo_demo.environment import ArithmeticEnv, Problem
from halo_demo.io import read_jsonl, trainable_groups, write_json


def run_stage(stage: str, run_dir: Path, round_index: int = 0, adapter: Path | None = None):
    command = [
        sys.executable,
        "-m",
        "halo_demo.worker",
        stage,
        "--run-dir",
        str(run_dir),
        "--round-index",
        str(round_index),
    ]
    if adapter:
        command += ["--adapter", str(adapter)]
    subprocess.run(command, check=True)


def train(args):
    config = load_config(args.config)
    overrides = {}
    for field in (
        "model_name",
        "model_revision",
        "dataset_name",
        "dataset_config",
        "dataset_revision",
        "dataset_train_split",
        "dataset_eval_split",
        "prompt_field",
        "answer_field",
        "answer_delimiter",
    ):
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    if "model_name" in overrides and "model_revision" not in overrides:
        overrides["model_revision"] = "main"
    if "dataset_name" in overrides and "dataset_revision" not in overrides:
        overrides["dataset_revision"] = "main"
    config = replace(config, **overrides)
    if args.smoke:
        config = replace(config, rounds=1, prompts_per_round=4, eval_size=4)
    # Fail before creating a run or downloading weights when CUDA/Halo is unavailable.
    subprocess.run([sys.executable, "-m", "halo_demo.worker", "doctor"], check=True)
    run_dir = Path(args.output or f"outputs/{datetime.now(timezone.utc):%Y%m%d-%H%M%S-%f}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", asdict(config))
    print(f"Run directory: {run_dir}", flush=True)
    subprocess.run([sys.executable, "-m", "halo_demo.data", "--run-dir", str(run_dir)], check=True)
    run_stage("evaluate", run_dir)
    adapter = None
    updated_rounds = 0
    for index in range(1, config.rounds + 1):
        print(f"\nRound {index}/{config.rounds}: collect → reward → Halo update", flush=True)
        run_stage("collect", run_dir, index, adapter)
        rows = read_jsonl(run_dir / f"round-{index:03d}" / "rollouts.jsonl")
        if trainable_groups(rows):
            run_stage("update", run_dir, index, adapter)
            adapter = run_dir / f"round-{index:03d}" / "adapter"
            updated_rounds += 1
        else:
            print("All groups have equal rewards; skipping this round's update.", flush=True)
    run_stage("evaluate", run_dir, config.rounds + 1, adapter)
    before = json.loads((run_dir / "eval-before.json").read_text())
    after = json.loads((run_dir / "eval-after.json").read_text())
    summary = {
        "updated_rounds": updated_rounds,
        "adapter": str(adapter.relative_to(run_dir)) if adapter else None,
        "before": before,
        "after": after,
        "accuracy_delta": after["accuracy"] - before["accuracy"],
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not updated_rounds:
        raise SystemExit(
            "No learning signal was collected and no adapter was trained. See rollouts.jsonl; try more prompts or generations."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Run sequential single-GPU RL")
    train_parser.add_argument("--config", default="configs/default.toml")
    train_parser.add_argument("--output", help="New output directory; existing paths are never overwritten")
    train_parser.add_argument("--smoke", action="store_true", help="One round, four prompts, four evaluation tasks")
    train_parser.add_argument("--model", dest="model_name", help="Hugging Face causal language model ID")
    train_parser.add_argument("--model-revision", help="Model commit/tag (defaults to main when --model changes)")
    train_parser.add_argument(
        "--dataset", dest="dataset_name", help="Hugging Face dataset ID; omitted uses config/synthetic tasks"
    )
    train_parser.add_argument("--dataset-config", help="Hugging Face dataset subset/config name")
    train_parser.add_argument("--dataset-revision", help="Dataset commit/tag")
    train_parser.add_argument("--dataset-train-split", help="Training split, e.g. train or train[:2048]")
    train_parser.add_argument("--dataset-eval-split", help="Optional separate evaluation split")
    train_parser.add_argument("--prompt-field", help="Dataset column with prompt text")
    train_parser.add_argument("--answer-field", help="Dataset column with the expected integer answer")
    train_parser.add_argument("--answer-delimiter", help="Extract the integer after the last delimiter, e.g. ####")
    commands.add_parser("doctor", help="Check CUDA, BF16, and the Halo API without downloading weights")
    commands.add_parser("verify", help="Run a real Halo optimizer/save/reload check on a tiny random model")
    commands.add_parser("demo", help="Show an environment episode without ML dependencies")
    args = parser.parse_args()
    if args.command == "demo":
        env = ArithmeticEnv()
        print(json.dumps(env.reset(Problem(7, "+", 5)), indent=2))
        print(env.step("<answer>12</answer>"))
    elif args.command == "doctor":
        subprocess.run([sys.executable, "-m", "halo_demo.worker", "doctor"], check=True)
    elif args.command == "verify":
        subprocess.run([sys.executable, "-m", "tests.gpu_check"], check=True)
    else:
        train(args)


if __name__ == "__main__":
    main()
