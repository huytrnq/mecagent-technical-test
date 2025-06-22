"""
Utility functions for CadQuery fine-tuning.
"""

import torch
from typing import List, Dict
from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info

from metrics.best_iou import evaluate_codes
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
import config

# === Data Processing Functions ===

def format_data(image: Image.Image, prompt: str, code: str) -> List[Dict]:
    """Format data for training."""
    return [
        {"role": "system", "content": "You are a helpful assistant. Return only valid CadQuery Python code."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": code.strip()}
        ]}
    ]

def load_train_dataset():
    """Load training dataset."""
    print(f"Loading training dataset ({config.NUM_TRAIN_SAMPLES} samples)...")
    dataset = load_dataset("CADCODER/GenCAD-Code", split="train", cache_dir=config.DATA_CACHE_DIR)
    dataset = dataset.select(range(config.NUM_TRAIN_SAMPLES))
    return [format_data(sample["image"], sample["prompt"], sample["cadquery"]) for sample in dataset]

def load_val_dataset():
    """Load validation dataset."""
    print(f"Loading validation dataset ({config.NUM_VAL_SAMPLES} samples)...")
    dataset = load_dataset("CADCODER/GenCAD-Code", split="test", cache_dir=config.DATA_CACHE_DIR)
    dataset = dataset.select(range(config.NUM_VAL_SAMPLES))
    
    # Formatted version for training
    formatted_dataset = [format_data(sample["image"], sample["prompt"], sample["cadquery"]) for sample in dataset]
    
    # Raw version for metrics
    raw_dataset = dataset
    
    return formatted_dataset, raw_dataset

# === Inference Functions ===

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    """Generate text from a formatted sample."""
    try:
        text_input = processor.apply_chat_template(sample[1:2], tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(sample)
        
        model_inputs = processor(
            text=[text_input],
            images=image_inputs,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        
        trimmed_generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]
    except Exception as e:
        return f"# ERROR: {e}"

# === Metrics Functions ===

def compute_metrics(model, processor, val_samples, max_eval_samples=None):
    """Compute VSR and IOU metrics."""
    if max_eval_samples is None:
        max_eval_samples = config.MAX_EVAL_SAMPLES
        
    print(f"\n=== Computing metrics on {min(max_eval_samples, len(val_samples))} samples ===")
    
    # Prepare samples
    if hasattr(val_samples, 'select'):
        eval_samples = val_samples.select(range(min(max_eval_samples, len(val_samples))))
        eval_samples = [eval_samples[i] for i in range(len(eval_samples))]
    else:
        eval_samples = val_samples[:max_eval_samples]
    
    gt_codes = {}
    pred_codes = {}
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(eval_samples):
            try:
                formatted_sample = format_data(sample["image"], sample["prompt"], sample["cadquery"])
                pred_code = generate_text_from_sample(model, processor, formatted_sample)
                
                gt_codes[str(i)] = sample["cadquery"]
                pred_codes[str(i)] = pred_code
                
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{len(eval_samples)} predictions")
                    
            except Exception as e:
                print(f"Error generating prediction for sample {i}: {e}")
                gt_codes[str(i)] = sample["cadquery"]
                pred_codes[str(i)] = f"# ERROR: {e}"
    
    # Calculate metrics
    vsr = evaluate_syntax_rate_simple(pred_codes)
    
    try:
        iou_results = evaluate_codes(gt_codes, pred_codes)
        iou_best = iou_results.get("iou_best", 0.0)
    except Exception as e:
        print(f"Error computing IoU: {e}")
        iou_best = 0.0
    
    print(f"Valid Syntax Rate: {vsr:.3f}")
    print(f"Mean IoU Best: {iou_best:.3f}")
    
    return {"eval_vsr": vsr, "eval_iou_best": iou_best}

# === Collate Function ===

def collate_fn(examples, processor):
    """Collate function for batching."""
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example)[0] for example in examples]

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask image tokens
    image_tokens = [151652, 151653, 151655]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels
    return batch