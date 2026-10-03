"""Real Halo update/save/reload check, using a tiny random model and no Hub download."""

import tempfile
from pathlib import Path


def main():
    from halo_demo.worker import check_runtime

    print(check_runtime())
    import torch
    from accelerate import PartialState
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from src.configs.offline_grpo_config import OfflineGRPOConfig
    from src.distributed.parallelism_config import ParallelismConfig
    from src.trainers.grpo.offline import OfflineGRPOTrainer
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    PartialState()
    torch.manual_seed(42)
    backend = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "What": 3, "is": 4, "2": 5, "+": 6, "?": 7, "4": 8, "5": 9},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    config = Qwen2Config(
        vocab_size=10,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=None,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    base = Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    # Preserve the exact random base to test that the saved adapter restores the same policy.
    base_state = {key: value.detach().cpu().clone() for key, value in base.state_dict().items()}
    model = get_peft_model(
        base, LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    )
    before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    with tempfile.TemporaryDirectory() as directory:
        trainer = OfflineGRPOTrainer(
            model=model,
            args=OfflineGRPOConfig(
                output_dir=directory,
                max_steps=1,
                per_device_train_batch_size=2,
                learning_rate=1e-3,
                lr_scheduler_type="constant",
                optim="adamw_torch",
                max_prompt_length=32,
                max_completion_length=8,
                bf16=True,
                advantage_method="z_norm",
                policy_gradient_formulation="reinforce",
                loss_type="grpo",
                remove_unused_columns=False,
                report_to="none",
                save_strategy="no",
                dataset_num_proc=1,
            ),
            train_dataset=Dataset.from_list(
                [{"prompt": "What is 2 + 2 ?", "completions": ["4 [EOS]", "5 [EOS]"], "rewards": [1.0, 0.0]}]
            ),
            processing_class=tokenizer,
            parallelism_config=ParallelismConfig(
                world_size=1,
                gpus_per_node=1,
                use_grouped_gemm=False,
                bf16_optimizer=False,
            ),
        )
        result = trainer.train()
        assert result.global_step == 1, "Halo did not take an optimizer step"
        assert torch.isfinite(torch.tensor(result.training_loss)), "Training loss is non-finite"
        assert any(
            not torch.equal(before[name], value) for name, value in model.named_parameters() if name in before
        ), "LoRA weights did not change"
        checkpoint = Path(directory) / "adapter"
        trainer.save_model(str(checkpoint))
        assert (checkpoint / "adapter_model.safetensors").is_file(), "Adapter was not saved"
        fresh_base = Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
        fresh_base.load_state_dict(base_state)
        restored = PeftModel.from_pretrained(fresh_base, checkpoint).eval()
        model.eval()
        probe = torch.tensor([[3, 4, 5, 6, 5, 7]], device="cuda")
        with torch.inference_mode():
            torch.testing.assert_close(model(probe).logits, restored(probe).logits)
    print("PASS: Halo took a real optimizer step, changed LoRA weights, and saved/reloaded the policy.")


if __name__ == "__main__":
    main()
