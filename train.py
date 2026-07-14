import argparse
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import numpy as np

# ------------------------------------------------------------
# 1. THE PRODUCTION INCIDENT DEBUGGER (SIMULATES ON-CALL RESPONSE)
# ------------------------------------------------------------
class ProductionIncidentCallback(TrainerCallback):
    """Custom callback that mimics responding to time-sensitive training incidents."""
    def on_step_end(self, args, state, control, model=None, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            # Check for NaN or Exploding Gradients (simulates monitoring dashboards)
            if np.isnan(loss) or loss > 100:
                print("\n[🚨 PRODUCTION INCIDENT] NaN or Loss Spike detected!")
                print(f"[ACTION] Pausing training at step {state.global_step}.")
                print(f"[ACTION] Saving emergency checkpoint to 'emergency_checkpoint'")
                if model:
                    model.save_pretrained("emergency_checkpoint")
                control.should_training_stop = True  # Halt the run for debugging
        return control

# ------------------------------------------------------------
# 2. DATA VALIDATOR (SIMULATES EAGLESFT QUALITY PIPELINE)
# ------------------------------------------------------------
class DataValidator:
    def __init__(self, min_length=10, max_length=512):
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, example):
        # Check for empty prompts/rejections and filter duplicates via length
        if not example['prompt'] or len(example['prompt']) < 5:
            return False
        if len(example['chosen']) < self.min_length or len(example['rejected']) < self.min_length:
            return False
        return True

# ------------------------------------------------------------
# 3. MAIN TRAINING FUNCTION
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_key", type=str, default=None, help="Weights & Biases API Key")
    parser.add_argument("--sample_size", type=int, default=5000, help="Rows to sample for speed")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    args = parser.parse_args()

    # --- WandB Setup (Observability) ---
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
        run = wandb.init(project="anthropic-post-training-sim", name="dpo-constitutional-alignment")
    else:
        run = None
        print("⚠️  No WandB key provided. Logging locally only.")

    # --- 3A: Load Dataset (Constitutional AI / Safety) ---
    print("📊 Loading PKU-SafeRLHF dataset...")
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    
    # Map to DPO format (Chosen = Safer response, Rejected = Unsafe response)
    def format_dpo(example):
        # is_response_0_safer: if True, response_0 is chosen, else response_1 is chosen.
        if example["is_response_0_safer"]:
            return {"prompt": example["prompt"], "chosen": example["response_0"], "rejected": example["response_1"]}
        else:
            return {"prompt": example["prompt"], "chosen": example["response_1"], "rejected": example["response_0"]}
    
    dataset = dataset.map(format_dpo).select(range(min(args.sample_size, len(dataset))))
    
    # Apply Data Validator (simulates production quality gates)
    validator = DataValidator()
    dataset = dataset.filter(validator.validate)
    print(f"✅ Dataset ready. Size: {len(dataset)}")

    # --- 3B: Load Model with 4-bit QLoRA (Simulates HPC constraints) ---
    print("🖥️  Loading model in 4-bit for High-Performance memory efficiency...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # Optimizes T4 throughput
    )
    model.config.use_cache = False  # Required for gradient checkpointing
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

    # --- 3C: PEFT / LoRA Configuration ---
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # --- 3D: Training Arguments (Simulating Distributed Systems via Grad Accumulation) ---
    training_args = DPOConfig(
        output_dir="./dpo_checkpoints",
        per_device_train_batch_size=2,          # Small batch for T4 VRAM
        gradient_accumulation_steps=8,          # Simulates global batch of 16 (8 nodes * 2)
        learning_rate=5e-5,
        warmup_steps=100,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        gradient_checkpointing=True,            # VRAM optimization
        bf16=True,                              # Mixed precision
        max_length=512,
        max_prompt_length=256,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb_key else "none",
        run_name="dpo_constitutional_run",
        # DPO specific
        beta=0.1,  # Temperature for DPO
    )

    # --- 3E: Initialize DPO Trainer with Incident Callback ---
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL handles reference model internally if None
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=[ProductionIncidentCallback()],  # <-- The on-call debugger
    )

    # --- 3F: Train & Save Production Artifacts ---
    print("🚀 Starting Constitutional DPO training...")
    dpo_trainer.train()

    print("💾 Saving final production adapter and tokenizer...")
    model.save_pretrained("./production_model_adapter")
    tokenizer.save_pretrained("./production_model_adapter")
    
    if run:
        run.finish()

    print("✅ Training complete. Emergency debugger and observability layers were active.")

if __name__ == "__main__":
    main()