"""
Simplified CadQuery fine-tuning script.
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import get_peft_model
from trl import SFTTrainer

import config
import utils

# === Custom Trainer ===

class CustomSFTTrainer(SFTTrainer):
    """Custom trainer with metrics computation."""
    
    def __init__(self, val_samples=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_samples = val_samples
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        if self.val_samples is not None:
            try:
                custom_metrics = utils.compute_metrics(
                    self.model, self.processing_class, self.val_samples, config.MAX_EVAL_SAMPLES
                )
                eval_results.update(custom_metrics)
            except Exception as e:
                print(f"Error computing custom metrics: {e}")
        
        return eval_results

# === Main Training Function ===

def main():
    """Main training pipeline."""
    print("CadQuery Fine-tuning")
    print("=" * 40)
    
    # Setup
    config.setup_gpu()
    
    # Load model and processor
    print(f"Loading model: {config.MODEL_ID}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=config.CACHE_DIR,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        config.MODEL_ID,
        trust_remote_code=True,
        cache_dir=config.CACHE_DIR
    )
    
    # Setup PEFT
    print("Setting up PEFT...")
    peft_config = config.get_peft_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load datasets
    train_dataset = utils.load_train_dataset()
    val_dataset, val_dataset_raw = utils.load_val_dataset()
    
    # Training arguments
    training_args = config.get_training_args()
    
    # Create trainer
    print("Initializing trainer...")
    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda examples: utils.collate_fn(examples, processor),
        peft_config=peft_config,
        processing_class=processor.tokenizer,
        val_samples=val_dataset_raw,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save
    print(f"Saving model to {config.OUTPUT_DIR}")
    trainer.save_model(config.OUTPUT_DIR)
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_metrics = utils.compute_metrics(
        trainer.model, processor, val_dataset_raw, config.FINAL_EVAL_SAMPLES
    )
    
    print(f"Final VSR: {final_metrics.get('eval_vsr', 0.0):.3f}")
    print(f"Final IoU Best: {final_metrics.get('eval_iou_best', 0.0):.3f}")
    print("✅ Training completed!")
    
    return final_metrics

if __name__ == "__main__":
    main()