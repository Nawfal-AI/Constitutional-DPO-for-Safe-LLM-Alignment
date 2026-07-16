import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import DPOConfig, DPOTrainer

# ------------------------------------------------------------
# 1. TRAINING HEALTH MONITOR
# ------------------------------------------------------------
# Watches loss for NaNs / spikes and checkpoints + halts on detection,
# similar in spirit to an automated on-call response to a bad training run.
class TrainingHealthCallback(TrainerCallback):
    def __init__(self, loss_spike_threshold: float = 100.0):
        self.loss_spike_threshold = loss_spike_threshold

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return control
        loss = logs["loss"]
        if np.isnan(loss) or np.isinf(loss) or loss > self.loss_spike_threshold:
            print(f"\n[ALERT] Anomalous loss detected at step {state.global_step}: {loss}")
            print("[ACTION] Saving emergency checkpoint to 'emergency_checkpoint/' and halting run.")
            if model is not None:
                model.save_pretrained("emergency_checkpoint")
            control.should_training_stop = True
        return control


# ------------------------------------------------------------
# 2. DATA VALIDATOR
# ------------------------------------------------------------
# Basic quality gate: drops empty, too-short, or malformed preference pairs
# before they reach the trainer.
class DataValidator:
    def __init__(self, min_length: int = 10):
        self.min_length = min_length

    def validate(self, example) -> bool:
        if not example.get("prompt") or len(example["prompt"]) < 5:
            return False
        if len(example.get("chosen", "")) < self.min_length:
            return False
        if len(example.get("rejected", "")) < self.min_length:
            return False
        if example["chosen"].strip() == example["rejected"].strip():
            return False
        return True


def pick_attn_implementation() -> str:
    """
    flash_attention_2 requires a GPU with compute capability >= 8.0 (Ampere+).
    Kaggle's free T4 is compute capability 7.5 and will error or silently
    fall back if you force flash-attention-2 there. This picks something
    that actually runs on the hardware you have.
    """
    if not torch.cuda.is_available():
        return "eager"
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8:
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            pass
    # sdpa (PyTorch's built-in scaled-dot-product-attention) runs on T4
    # and is meaningfully faster than eager without the Ampere requirement.
    return "sdpa"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_key", type=str, default=None, help="Weights & Biases API key")
    parser.add_argument("--sample_size", type=int, default=5000, help="Rows to sample for speed")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--output_dir", type=str, default="./dpo_checkpoints")
    parser.add_argument("--adapter_dir", type=str, default="./production_model_adapter")
    parser.add_argument(
        "--effective_batch_size",
        type=int,
        default=16,
        help="per_device_train_batch_size * gradient_accumulation_steps",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument(
        "--disable_gradient_checkpointing",
        action="store_true",
        help="Trade VRAM for speed. With LoRA (~9M trainable params on a 4-bit "
        "base model) there's usually enough headroom on a 16GB T4 to disable "
        "this and get a meaningful speedup.",
    )
    # parse_known_args (not parse_args) so this doesn't crash when run inside
    # a notebook kernel (Colab/Jupyter inject their own "-f kernel.json" arg
    # that argparse doesn't recognize).
    args, _unknown = parser.parse_known_args()

    grad_accum_steps = max(1, args.effective_batch_size // args.per_device_train_batch_size)

    run = None
    if args.wandb_key:
        import wandb

        wandb.login(key=args.wandb_key)
        run = wandb.init(project="dpo-safety-alignment", name="phi3-dpo-run")
    else:
        print("No W&B key provided — logging to console/JSON only.")

    # --- Load and format the preference dataset ---
    print("Loading PKU-SafeRLHF dataset...")
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")

    def format_dpo(example):
        # The dataset schema uses `safer_response_id` (0 or 1) to indicate
        # which response is safer. Older dataset snapshots used a boolean
        # `is_response_0_safer` column instead — handle both so this keeps
        # working if you pull a different revision.
        if "safer_response_id" in example:
            safer_is_0 = example["safer_response_id"] == 0
        elif "is_response_0_safer" in example:
            safer_is_0 = example["is_response_0_safer"]
        else:
            raise KeyError(
                "Neither 'safer_response_id' nor 'is_response_0_safer' found in this "
                "dataset example — the PKU-SafeRLHF schema may have changed again. "
                f"Available columns: {list(example.keys())}"
            )

        if safer_is_0:
            return {"prompt": example["prompt"], "chosen": example["response_0"], "rejected": example["response_1"]}
        return {"prompt": example["prompt"], "chosen": example["response_1"], "rejected": example["response_0"]}

    dataset = dataset.map(format_dpo).select(range(min(args.sample_size, len(dataset))))

    validator = DataValidator()
    pre_filter_size = len(dataset)
    dataset = dataset.filter(validator.validate)
    print(f"Dataset ready: {len(dataset)} / {pre_filter_size} examples passed validation.")

    # --- Load model in 4-bit (QLoRA) to fit a ~3.8B model on 16GB VRAM ---
    attn_impl = pick_attn_implementation()
    print(f"Loading model in 4-bit (attn implementation: {attn_impl})...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        # NOT trust_remote_code=True: Phi-3's cached remote modeling code
        # expects the old rope_scaling schema (key "type") while current
        # transformers normalizes configs to the newer schema (key
        # "rope_type"), causing `KeyError: 'type'`. transformers has had
        # native Phi-3 support since 4.41, which stays in sync with itself
        # and also correctly supports "sdpa" (the remote code doesn't).
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training config ---
    # Note: gradient_accumulation_steps increases the *effective batch size*
    # on a single device. It is not a substitute for multi-GPU / multi-node
    # training (which would require e.g. DeepSpeed or FSDP with accelerate
    # launch across multiple devices) — it just changes how often we step
    # the optimizer.
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=5e-5,
        warmup_steps=100,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=25,
        save_total_limit=2,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
        bf16=torch.cuda.is_available(),
        max_length=512,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb_key else "none",
        run_name="dpo_safety_alignment_run",
        beta=0.1,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL builds the reference model internally when None
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[TrainingHealthCallback()],
    )

    print("Starting DPO training...")
    train_result = dpo_trainer.train()

    print(f"Saving adapter and tokenizer to {args.adapter_dir}...")
    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)

    # Persist real metrics from this run instead of hardcoding numbers anywhere.
    metrics = dict(train_result.metrics)
    metrics["effective_batch_size"] = args.effective_batch_size
    metrics["attn_implementation"] = attn_impl
    metrics["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    os.makedirs("results", exist_ok=True)
    with open("results/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved training metrics to results/train_metrics.json")

    if run:
        run.finish()

    print("Training complete.")


if __name__ == "__main__":
    main()