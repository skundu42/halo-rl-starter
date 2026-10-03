"""One GPU phase per process: releasing the process releases its CUDA allocations."""

import argparse
import importlib.metadata
import json
import random
import subprocess
from pathlib import Path

from halo_demo.config import Config
from halo_demo.data import read_tasks
from halo_demo.environment import ArithmeticEnv
from halo_demo.io import read_jsonl, trainable_groups, write_json, write_jsonl


def check_runtime() -> dict:
    try:
        # Halo installs its kernel dispatch shim before transformers models are imported.
        import src  # noqa: F401
        import torch
        from src.configs.offline_grpo_config import OfflineGRPOConfig  # noqa: F401
        from src.trainers.grpo.offline import OfflineGRPOTrainer  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            f"Halo runtime is unavailable: {error}. Use ./run.sh to run the pinned Docker image."
        ) from error
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run on Linux with an NVIDIA GPU and NVIDIA Container Toolkit.")
    if torch.cuda.device_count() != 1:
        raise SystemExit("This demo needs exactly one visible GPU. Use GPU_DEVICE=0 ./run.sh train.")
    if torch.cuda.get_device_capability(0)[0] < 8 or not torch.cuda.is_bf16_supported():
        raise SystemExit("This BF16 recipe requires an Ampere or newer NVIDIA GPU.")
    properties = torch.cuda.get_device_properties(0)
    # A real allocation catches driver/runtime failures before downloading the model.
    probe = torch.ones(16, device="cuda", dtype=torch.bfloat16)
    _ = (probe @ probe).item()
    halo_root = Path(src.__file__).resolve().parent.parent
    revision = subprocess.run(
        ["git", "-c", f"safe.directory={halo_root}", "-C", str(halo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "gpu": properties.name,
        "vram_gib": round(properties.total_memory / 1024**3, 1),
        "cuda": torch.version.cuda,
        "halo_revision": revision,
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "trl", "halo")
        },
    }


def load_policy(config: Config, adapter: str | None, *, training: bool):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision)
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError("The model must have a chat template and EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=training)
    elif training:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                revision=config.model_revision,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=0.0,
                target_modules=config.lora_target_modules,
            ),
        )
    model.config.use_cache = not training
    if training:
        model.print_trainable_parameters()
    else:
        model.eval()
    return model, tokenizer


def generate(model, tokenizer, problem, config: Config, count: int, *, sample: bool):
    import torch
    from src.data.spans import resolve_eos_token_ids
    from transformers import GenerationConfig

    prompt = tokenizer.apply_chat_template(
        problem.messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=config.enable_thinking,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    if inputs.input_ids.shape[1] > config.max_prompt_tokens:
        raise ValueError(
            f"Rendered prompt exceeds max_prompt_tokens={config.max_prompt_tokens}; increase it in your config."
        )
    # Unwarped samples match the raw policy log-probs used by Halo's offline objective.
    eos_ids = resolve_eos_token_ids(tokenizer, model.config)
    generation = GenerationConfig(
        do_sample=sample,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        max_new_tokens=config.max_new_tokens,
        num_return_sequences=count,
        eos_token_id=sorted(eos_ids),
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    with torch.inference_mode():
        sequences = model.generate(**inputs, generation_config=generation)
    responses = []
    for sequence in sequences[:, inputs.input_ids.shape[1] :].tolist():
        stop_index = next((index for index, token in enumerate(sequence) if token in eos_ids), None)
        terminated = stop_index is not None
        if terminated:
            sequence = sequence[: stop_index + 1]
        text = tokenizer.decode(sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # Keep the actual EOS in stored completions; Halo must not learn padding tokens.
        completion = tokenizer.decode(sequence, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        env = ArithmeticEnv()
        env.reset(problem)
        _, _, _, info = env.step(text, truncated=not terminated)
        responses.append({"text": text, "completion": completion, **info})
    return prompt, responses


def summarize(responses: list[dict]) -> dict:
    count = len(responses)
    return {
        "samples": count,
        "accuracy": sum(r["correct"] for r in responses) / count,
        "format_rate": sum(r["formatted"] for r in responses) / count,
        "mean_reward": sum(r["reward"] for r in responses) / count,
        "truncation_rate": sum(r["truncated"] for r in responses) / count,
    }


def collect(config: Config, run_dir: Path, round_index: int, adapter: str | None):
    from transformers import set_seed

    model, tokenizer = load_policy(config, adapter, training=False)
    set_seed(config.seed + round_index)
    pool = read_tasks(run_dir, "train")
    problems = random.Random(config.seed + round_index).sample(pool, config.prompts_per_round)
    rows, all_responses = [], []
    round_dir = run_dir / f"round-{round_index:03d}"
    round_dir.mkdir()
    for index, problem in enumerate(problems, 1):
        prompt, responses = generate(model, tokenizer, problem, config, config.num_generations, sample=True)
        rows.append(
            {
                "prompt": prompt,
                "completions": [r["completion"] for r in responses],
                "rewards": [r["reward"] for r in responses],
                "question": problem.prompt,
                "answer": problem.answer,
                "responses": responses,
            }
        )
        all_responses.extend(responses)
        print(f"Collected {index}/{len(problems)} prompts", flush=True)
    write_jsonl(round_dir / "rollouts.jsonl", rows)
    metrics = {
        **summarize(all_responses),
        "trainable_groups": len(trainable_groups(rows)),
        "total_groups": len(rows),
    }
    write_json(round_dir / "rollout-metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def update(config: Config, run_dir: Path, round_index: int, adapter: str | None):
    from accelerate import PartialState
    from datasets import Dataset
    from src.configs.offline_grpo_config import OfflineGRPOConfig
    from src.distributed.parallelism_config import ParallelismConfig
    from src.trainers.grpo.offline import OfflineGRPOTrainer

    PartialState()
    round_dir = run_dir / f"round-{round_index:03d}"
    rows = trainable_groups(read_jsonl(round_dir / "rollouts.jsonl"))
    if not rows:
        raise ValueError("No groups with unequal rewards; there is no learning signal.")
    model, tokenizer = load_policy(config, adapter, training=True)
    training_args = OfflineGRPOConfig(
        output_dir=str(round_dir / "trainer"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_prompt_length=config.max_prompt_tokens,
        max_completion_length=config.max_new_tokens,
        advantage_method="z_norm",
        loss_type="grpo",
        policy_gradient_formulation="reinforce",
        kl_beta=0.0,
        min_log_prob=-3.0,
        drop_degenerate_groups=True,
        dataset_num_proc=1,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        logging_steps=1,
        seed=config.seed + round_index,
    )
    # Programmatic Halo API expects rendered strings; the Halo CLI renders message lists.
    dataset = Dataset.from_list([{key: row[key] for key in ("prompt", "completions", "rewards")} for row in rows])
    trainer = OfflineGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        parallelism_config=ParallelismConfig(
            world_size=1,
            gpus_per_node=1,
            use_grouped_gemm=False,
            bf16_optimizer=False,
        ),
    )
    result = trainer.train()
    trainer.save_model(str(round_dir / "adapter"))
    tokenizer.save_pretrained(round_dir / "adapter")
    write_json(round_dir / "training-metrics.json", result.metrics)


def evaluate(config: Config, run_dir: Path, round_index: int, adapter: str | None):
    model, tokenizer = load_policy(config, adapter, training=False)
    problems = read_tasks(run_dir, "eval")
    records = []
    for index, problem in enumerate(problems, 1):
        _, responses = generate(model, tokenizer, problem, config, 1, sample=False)
        records.append({"question": problem.prompt, "answer": problem.answer, **responses[0]})
        print(f"Evaluated {index}/{len(problems)} prompts", flush=True)
    label = "before" if round_index == 0 else "after"
    write_jsonl(run_dir / f"eval-{label}.jsonl", records)
    write_json(run_dir / f"eval-{label}.json", summarize(records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("doctor", "collect", "update", "evaluate"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--adapter")
    args = parser.parse_args()
    runtime = check_runtime()
    if args.stage == "doctor":
        print(json.dumps(runtime, indent=2))
        return
    if args.run_dir is None:
        parser.error("--run-dir is required for GPU stages")
    config = Config(**json.loads((args.run_dir / "config.json").read_text()))
    write_json(args.run_dir / "runtime.json", runtime)
    {"collect": collect, "update": update, "evaluate": evaluate}[args.stage](
        config,
        args.run_dir,
        args.round_index,
        args.adapter,
    )


if __name__ == "__main__":
    main()
