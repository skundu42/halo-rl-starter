# Halo: a small, single-GPU RL environment

Train a Hugging Face language model to answer arithmetic questions using **White Circle's [Halo](https://github.com/whitecircle/halo)**. Clone this project on a Linux NVIDIA machine, then run the Docker launcher. Defaults target **one GPU with 24 GB or more VRAM**, using `Qwen/Qwen2.5-0.5B-Instruct` and LoRA.

Both the **model and dataset are configurable from Hugging Face**, through CLI flags or TOML. The default dataset is generated locally, so no dataset account or external judge is needed.

## Quick start

On the GPU machine, install Git, Docker, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), and a current NVIDIA driver. The Halo images use CUDA 13; NVIDIA lists driver **580 or newer** as its compatibility floor. Use a driver supported by your GPU and the image's CUDA runtime. [CUDA compatibility reference](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

```bash
git clone https://github.com/skundu42/halo-rl-starter.git
cd halo-rl-starter

./run.sh doctor          # builds/pulls once, checks Halo + CUDA + BF16
./run.sh verify          # real tiny-model optimizer/save/reload test; no model download
./run.sh train --smoke   # one round with the actual HF model
./run.sh train           # five rounds, 32 prompts × 4 samples per round
```

The first build pulls Halo's large prebuilt training image and downloads the model on first use. Keep ample disk space for the image, cache, and outputs. No host Python installation, vLLM service, W&B login, or API key is required.

For **H100/H200**, use `HALO_ARCH=hopper ./run.sh train`. For **B200/B300**, keep the default `blackwell` image. Ampere/Ada cards such as RTX 3090/4090 use the default image with SDPA; upstream labels that consumer-GPU path unvalidated. This project also needs a GPU validation run; no NVIDIA hardware was available during development. The official images are Linux **x86_64**, not ARM64/Grace or macOS. [Halo installation matrix](https://github.com/whitecircle/halo/blob/0bc3a22a56fb5a7a422b1d4f711d92a978a990bd/human-docs/installation.md).

## Pick a Hugging Face model and dataset

Use the included GSM8K config:

```bash
./run.sh train --config configs/gsm8k.toml
```

Or override the model and dataset directly:

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

`--model-revision` and `--dataset-revision` accept Hub commits or tags. Changing the ID on the CLI defaults its revision to `main`; moving refs are resolved to a commit once at startup and recorded in `config.json`. The shipped default model and GSM8K config pin commits. In a TOML file, update the revision when changing the associated ID.

For another dataset, copy `configs/gsm8k.toml` to `configs/custom.toml` and edit:

```toml
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model_revision = "main"
dataset_name = "openai/gsm8k"
dataset_config = "main"
dataset_revision = "main"
dataset_train_split = "train"
dataset_eval_split = "test"
prompt_field = "question"
answer_field = "answer"
answer_delimiter = "####"
dataset_train_limit = 2048
max_prompt_tokens = 512
max_new_tokens = 64
lora_target_modules = ["q_proj", "v_proj"]
```

Omit `dataset_config` if the dataset has no subset. Omit `answer_delimiter` when the answer column already contains integers. Omit `dataset_eval_split` to reserve a deterministic held-out sample from the training split. Exact duplicate prompts are removed, including overlap with the evaluation sample. Each run freezes the normalized train/eval rows and records dataset fingerprints in `data/`.

**Dataset contract:** a string prompt and an integer answer (up to nine digits), optionally extracted after a delimiter. GSM8K works because its answer ends in `#### NUMBER`. Floats, multiple-choice labels, free-form answers, and tool-use episodes require replacing `normalize_row()` and `score()`; selecting an arbitrary HF dataset does not automatically supply a correct reward function. Invalid schemas and non-integer answers fail with a row-specific error.

**Model contract:** a Transformers causal LM with a chat template, EOS token, BF16/SDPA support, and PEFT-compatible target modules. LoRA targets are configurable for architectures with different names. `enable_thinking = false` is passed to templates that support it. Larger models and longer contexts need more VRAM; a different HF ID is not a guarantee it fits 24 GB. Models requiring remote Python code are not enabled.

Private/gated Hub resources use your existing `HF_TOKEN` environment variable, forwarded into Docker. Export it in your shell; do not put it in a tracked config.

## What is being trained

```text
prompt → sample 4 answers → environment reward
                               ↓
                    group-relative advantages
                               ↓
                 Halo OfflineGRPOTrainer + LoRA
                               ↓
                   updated policy → next round
```

`ArithmeticEnv.reset(task)` returns the prompt. `step(answer)` terminates the episode and rewards:

| Response | Reward |
| --- | ---: |
| Exactly `<answer>12</answer>` when the answer is 12 | 1.0 |
| Correct format, wrong integer | 0.1 |
| Malformed, extra prose/answers, or generation truncated before EOS | 0.0 |

This is a **one-step RL environment**, also called a contextual bandit. The solution is kept in the grader, never inserted into the model prompt. Groups whose rewards all tie have zero advantage and are skipped. If every round ties, the command exits unsuccessfully and explicitly reports that no adapter was trained.

Halo's native online/environmental GRPO modes require separate trainer and rollout GPUs. This demo instead collects fresh samples, performs one epoch with Halo's **offline GRPO objective**, and repeats sequentially on one GPU. It is not Halo's asynchronous online trainer or PPO-clipped online GRPO. [Halo online GPU requirement](https://github.com/whitecircle/halo/blob/0bc3a22a56fb5a7a422b1d4f711d92a978a990bd/human-docs/training-methods/online-grpo.md).

The settings use group z-normalization, the `reinforce` policy-gradient formulation, a negative log-probability floor, and no KL reference model. Each phase runs in a fresh process to release GPU memory. LoRA weights carry across rounds; **optimizer state resets each round**. Samples use temperature 1 without top-k/top-p filtering. Learning is not guaranteed: this is a compact demonstration, and rewards or held-out accuracy can decline.

## Outputs and evaluation

Each invocation creates a new directory under `outputs/`:

```text
config.json                  effective settings
runtime.json                 GPU, library versions, Halo source revision
data/                        frozen task splits and source metadata
eval-before.json[l]           greedy baseline metrics and answers
round-001/rollouts.jsonl      grouped completions, ground truths, rewards
round-001/rollout-metrics.json
round-001/training-metrics.json
round-001/adapter/            PEFT adapter + tokenizer, when an update occurred
...
eval-after.json[l]            same held-out tasks, after training
summary.json                 accuracy change and latest adapter path
```

Compare `before.accuracy` and `after.accuracy` in `summary.json`; format reward alone is not task success. `trainable_groups` shows whether a round produced learning signal. The adapter path is relative to the run directory. Load it on the same base model/revision with `PeftModel.from_pretrained(base_model, adapter_path)`. Checkpoints are adapters, not standalone merged models. Run directories are never overwritten; automatic interrupted-run resume is not implemented.

```bash
GPU_DEVICE=1 ./run.sh train                       # select one host GPU
CACHE_DIR=/mnt/hf-cache ./run.sh train            # move persistent downloads
./run.sh train --output outputs/my-experiment     # must be a new directory
REBUILD=1 ./run.sh train                          # after editing Python or Dockerfile
```

Configs are mounted live; code is baked into the image. Keep custom configs under `configs/` and outputs under `outputs/` so they are mounted and persisted. The container runs with your host UID/GID, so output files remain owned by you.

## Validation and extension

Without Docker or a GPU, Python 3.12+ can run the environment and tests:

```bash
python3 -m halo_demo demo
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

CPU tests cover grading, reward abuse cases, train/eval separation, dataset conversion, config validation, and round-to-round adapter selection. They do **not** prove CUDA compatibility. `./run.sh verify` checks that the real Halo trainer changes weights, saves a SafeTensors adapter, and reproduces the trained policy when reloaded. Follow with `train --smoke` to exercise Hub loading, sampling, scoring, and learning end to end.

To add a task, start with `halo_demo/environment.py` (reward) and `halo_demo/data.py` (input schema). `halo_demo/worker.py` contains the actual Halo trainer integration. Hyperparameters live in `configs/default.toml` and `halo_demo/config.py`.

If you run out of memory, reduce `num_generations`, prompt/completion lengths, or model size; keep at least two generations. If all groups tie, inspect the saved responses, increase the number of prompts/generations, or adjust task difficulty. If output is truncated, increase `max_new_tokens`. A high score on trivial arithmetic can leave no room to learn; GSM8K is harder and the tiny default model may instead fail most problems.

The Dockerfile uses Halo's versioned training image and pins its source commit. It deliberately avoids installing the unrelated PyPI package named `halo`. Upstream Halo retains its own [license](https://github.com/whitecircle/halo/blob/0bc3a22a56fb5a7a422b1d4f711d92a978a990bd/LICENSE); model and dataset licenses also remain applicable.
