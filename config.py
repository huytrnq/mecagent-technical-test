"""
Simple configuration for CadQuery fine-tuning.
"""

import os
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig

# === Global Configuration ===
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
CACHE_DIR = "/storage/Users/huy/caches"
DATA_CACHE_DIR = "data"
NUM_TRAIN_SAMPLES = 1000
NUM_VAL_SAMPLES = 100
OUTPUT_DIR = "Cadquery_2"

# Training parameters
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4

# Evaluation parameters
MAX_EVAL_SAMPLES = 20
FINAL_EVAL_SAMPLES = 100

# Hardware
GPU_ID = "0"

def setup_gpu():
    """Setup GPU configuration."""
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

def get_bnb_config():
    """Get quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16"
    )

def get_peft_config():
    """Get PEFT config."""
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        task_type="CAUSAL_LM",
    )

def get_training_args():
    """Get training arguments."""
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        logging_steps=10,
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        metric_for_best_model="eval_loss",
        greater_is_better=True,
        load_best_model_at_end=True,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        push_to_hub=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    training_args.remove_unused_columns = False
    return training_args