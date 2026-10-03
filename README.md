# Halo RL Starter

Train a Hugging Face language model with [Halo](https://github.com/whitecircle/halo) on a single NVIDIA GPU. Models and datasets are configurable; the default uses `Qwen/Qwen2.5-0.5B-Instruct`, LoRA, and generated arithmetic questions.

## Requirements

- Linux x86_64 with an Ampere or newer NVIDIA GPU; defaults target **24 GB+ VRAM**.
- Git, Docker, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- An NVIDIA driver compatible with CUDA 13 (**580 or newer**).
- Disk space for the Halo image, model downloads, and training outputs.

Use the default image for B200/B300 and Ampere/Ada cards. For H100/H200, prefix commands with `HALO_ARCH=hopper`. See [Halo's GPU compatibility notes](https://github.com/whitecircle/halo/blob/0bc3a22a56fb5a7a422b1d4f711d92a978a990bd/human-docs/installation.md) for hardware support.

## Quick start

```bash
git clone https://github.com/skundu42/halo-rl-starter.git
cd halo-rl-starter

./run.sh doctor          # build the image and check the GPU runtime
./run.sh train --smoke   # short trial run
./run.sh train           # full default run
```

The first run downloads the Docker image and model. No host Python setup or external inference server is required. The default run uses five rounds, with 32 questions and four sampled answers per question in each round.

## Choose a model and dataset

Use the included GSM8K configuration:

```bash
./run.sh train --config configs/gsm8k.toml
```

Or set Hugging Face IDs and dataset columns directly:

```bash
./run.sh train \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-train-split train \
  --dataset-eval-split test \
  --prompt-field question \
  --answer-field answer \
  --answer-delimiter '####'
```

The dataset must contain **text questions and integer answers**. `answer_delimiter` extracts the integer after a marker such as GSM8K's `####`; omit it when the answer column already contains integers. Other answer types require a different grader.

Models must support Transformers causal language modeling, a chat template, BF16/SDPA, and LoRA. Larger models and longer sequences need more VRAM. Export `HF_TOKEN` in your shell when using private or gated models and datasets.

## Configuration

Copy a file from `configs/`, edit it, and pass it with `--config`:

```bash
cp configs/gsm8k.toml configs/custom.toml
./run.sh train --config configs/custom.toml
```

| Setting | Purpose |
| --- | --- |
| `model_name`, `model_revision` | Hugging Face model ID and revision |
| `dataset_name`, `dataset_revision` | Hugging Face dataset ID and revision; omit the name for generated arithmetic |
| `dataset_config` | Dataset subset; omit if not needed |
| `dataset_train_split`, `dataset_eval_split` | Dataset splits; omit the evaluation split to hold out training rows |
| `prompt_field`, `answer_field`, `answer_delimiter` | Question and answer extraction |
| `dataset_train_limit`, `eval_size` | Number of training and evaluation tasks to load |
| `rounds`, `prompts_per_round`, `num_generations` | Training rounds, questions per round, and answers per question |
| `learning_rate`, `gradient_accumulation_steps` | Optimizer settings |
| `max_prompt_tokens`, `max_new_tokens` | Prompt and generated-answer token limits |
| `lora_rank`, `lora_alpha`, `lora_target_modules` | LoRA settings; adjust target modules for your model architecture |

When changing a model or dataset ID in TOML, also update its revision or set it to `"main"`. CLI ID overrides default to `main`; `--model-revision` and `--dataset-revision` select a specific commit or tag. Each run records the resolved commits and saves its dataset splits.

Keep custom configs under `configs/` and outputs under `outputs/` so Docker mounts persist them.

## How training works

The model sees each question and generates several answers. A Python grader compares them with the dataset's correct answer, which is kept out of the model prompt.

| Answer | Reward |
| --- | ---: |
| Correct integer in `<answer>INTEGER</answer>` format | 1.0 |
| Correct format, wrong integer | 0.1 |
| Malformed or truncated response | 0.0 |

Halo's **offline GRPO trainer** updates the LoRA adapter using these rewards. The next round samples fresh answers from the updated model. Generation and training run sequentially on one GPU; adapter weights carry across rounds, while optimizer state resets each round.

## Outputs

Each run creates a directory under `outputs/` containing:

- `summary.json`: before/after accuracy and the latest adapter path.
- `config.json` and `runtime.json`: settings, revisions, and runtime information.
- `data/`: saved training and evaluation tasks.
- `eval-before.jsonl` and `eval-after.jsonl`: answers on the same held-out questions.
- `round-*/`: sampled answers, rewards, training metrics, and saved LoRA adapters.

Load an adapter with `PeftModel.from_pretrained(base_model, adapter_path)`, using the same base model and revision. The adapter path in `summary.json` is relative to the run directory. Existing run directories are never overwritten; automatic resume is not supported.

## Useful commands

```bash
GPU_DEVICE=1 ./run.sh train                    # choose a GPU
HALO_ARCH=hopper ./run.sh train                # use H100/H200
CACHE_DIR=/mnt/hf-cache ./run.sh train          # choose a download cache
./run.sh train --output outputs/my-experiment  # choose a new output directory
REBUILD=1 ./run.sh train                       # rebuild after code changes
```

Config edits take effect immediately; Python and Dockerfile changes require a rebuild.

If memory runs out, use a smaller model, shorter sequences, or fewer generations (minimum two). If responses are truncated, increase `max_new_tokens`. Groups with identical rewards are skipped; if no updates occur, inspect the saved answers and adjust task difficulty or increase sampling.
