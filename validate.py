"""
Improved evaluation script for fine-tuned CadQuery model.
"""

import torch
from transformers import AutoProcessor
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration
from datasets import load_dataset

import config
import utils
from metrics.best_iou import get_iou_best, evaluate_codes
from metrics.valid_syntax_rate import evaluate_syntax_rate

# === Configuration ===
ADAPTER_PATH = "Cadquery_2/checkpoint-125"
NUM_SAMPLES = 20

def load_finetuned_model():
    """Load fine-tuned model with adapter."""
    print("Loading fine-tuned model...")
    
    # Load base model
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=config.CACHE_DIR,
    )
    
    # Load PEFT adapter
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        torch_dtype=torch.bfloat16,
    )
    
    model.eval()
    print("Fine-tuned model loaded successfully")
    return model

def load_test_dataset():
    """Load test dataset."""
    print(f"Loading test dataset ({NUM_SAMPLES} samples)...")
    dataset = load_dataset("CADCODER/GenCAD-Code", split="test", cache_dir=config.DATA_CACHE_DIR)
    dataset = dataset.select(range(NUM_SAMPLES))
    print("Test dataset loaded")
    return dataset

def run_inference(model, processor, dataset):
    """Run inference on test dataset."""
    print("Running inference...")
    
    gt_codes = {}
    pred_codes = {}
    successful_samples = []
    failed_samples = []
    
    for i, sample in enumerate(dataset):
        try:
            # Format sample for inference
            formatted_sample = utils.format_data(sample["image"], sample["prompt"], sample["cadquery"])
            
            # Generate prediction
            prediction = utils.generate_text_from_sample(model, processor, formatted_sample)
            
            # Store results
            gt_codes[str(i)] = sample["cadquery"]
            pred_codes[str(i)] = prediction
            
            print(f"Sample {i+1}/{NUM_SAMPLES} completed")
            
            # Try to compute individual IoU
            try:
                iou = get_iou_best(sample["cadquery"], prediction)
                successful_samples.append((i, iou))
            except Exception as e:
                failed_samples.append((i, str(e)))
                
        except Exception as e:
            print(f"Error generating prediction for sample {i}: {e}")
            gt_codes[str(i)] = sample["cadquery"]
            pred_codes[str(i)] = f"# ERROR: {e}"
            failed_samples.append((i, str(e)))
    
    return gt_codes, pred_codes, successful_samples, failed_samples

def evaluate_results(gt_codes, pred_codes):
    """Evaluate the results using VSR and IoU metrics."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Valid Syntax Rate
    print("\n📊 Valid Syntax Rate:")
    result = evaluate_syntax_rate(pred_codes)
    vsr = result['vsr']
    print(f"   VSR: {vsr:.3f} ({vsr*100:.1f}%)")
    
    # IoU Evaluation
    print("\nIoU Evaluation:")
    try:
        iou_results = evaluate_codes(gt_codes, pred_codes)
        iou_best = iou_results.get('iou_best', 0.0)
        print(f"   Mean IoU Best: {iou_best:.3f}")
    except Exception as e:
        print(f" Error computing IoU: {e}")
        iou_best = 0.0
    
    return vsr, iou_best

def analyze_samples(successful_samples, failed_samples, pred_codes, gt_codes):
    """Analyze individual sample results."""
    print("\nIndividual Sample Analysis:")
    print("-" * 40)
    
    print(f"Successful predictions: {len(successful_samples)}")
    print(f"Failed predictions: {len(failed_samples)}")
    
    # Top performing samples
    if successful_samples:
        successful_samples.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop 3 predictions by IoU:")
        for sample_id, iou in successful_samples[:3]:
            print(f"   Sample {sample_id}: IoU = {iou:.3f}")
    
    # Failed samples
    if failed_samples:
        print(f"\nailed predictions (first 3):")
        for sample_id, error in failed_samples[:3]:
            print(f"   Sample {sample_id}: {error[:100]}...")
    
    # Show example predictions
    print(f"\nExample Predictions:")
    for i, (sample_id, pred_code) in enumerate(list(pred_codes.items())[:2]):
        print(f"\n--- Sample {sample_id} ---")
        print("Generated Code:")
        lines = pred_code.split('\n')[:5]
        for line in lines:
            print(f"  {line}")
        if len(pred_code.split('\n')) > 5:
            total_lines = len(pred_code.split('\n'))
            remaining_lines = total_lines - 5
            print(f"  ... ({remaining_lines} more lines)")

def main():
    """Main evaluation pipeline."""
    print("Fine-tuned Model Evaluation")
    print("=" * 50)
    
    # Setup
    config.setup_gpu()
    
    # Load model and processor
    model = load_finetuned_model()
    processor = AutoProcessor.from_pretrained(config.MODEL_ID, trust_remote_code=True, cache_dir=config.CACHE_DIR)
    
    # Load test dataset
    dataset = load_test_dataset()
    
    # Run inference
    gt_codes, pred_codes, successful_samples, failed_samples = run_inference(model, processor, dataset)
    
    # Evaluate results
    vsr, iou_best = evaluate_results(gt_codes, pred_codes)
    
    # Analyze samples
    analyze_samples(successful_samples, failed_samples, pred_codes, gt_codes)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Samples Evaluated: {NUM_SAMPLES}")
    print(f"Valid Syntax Rate: {vsr:.3f} ({vsr*100:.1f}%)")
    print(f"Mean IoU Best: {iou_best:.3f}")
    print(f"Successful Samples: {len(successful_samples)}/{NUM_SAMPLES}")
    print("="*60)
    
    return {
        'vsr': vsr,
        'iou_best': iou_best,
        'successful_count': len(successful_samples),
        'total_samples': NUM_SAMPLES
    }

if __name__ == "__main__":
    results = main()