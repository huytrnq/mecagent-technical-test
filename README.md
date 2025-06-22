CADQuery Code Generator — Technical Assessment Report
Task Summary

The objective was to develop a model that generates CadQuery code from rendered images of 3D objects. The challenge lies in grounding the visual representation in precise geometric and syntactic structures that match a CAD programming language.

Due to time and hardware constraints, I chose to implement a focused experimental setup: fine-tuning a vision-language model (Qwen-VL-3B) on a 1k-sample subset of the GenCAD-Code dataset, validating on 10 examples, and assessing quality through code and shape-based metrics.

The reasons I selected Qwen-VL-3B-Instruct are: 
- Its instruction-following nature increases the chance of outputting structured, well-formatted CadQuery code.
- With 3 billion parameters and 4-bit quantization, I could run it on a single 24 GB GPU.

Dataset and Preprocessing
The dataset used was CADCODER/GenCAD-Code, which contains over 147,000 pairs of:
•	an image 
•	a language prompt for CadQuery code generation
•	a CadQuery code snippet corresponding to the image

Due to computational limitations, I randomly sampled 1,000 pairs from the dataset for training and 10 pairs for validation. Because the validation steps involve mesh generation for iou calaulation so it tool a long time to run, therefore I only used 20 pairs for validation. Despite the small size, this subset was sufficient to demonstrate the model's capabilities when the generated code is syntactically correct, however, it is not enough to capture the shape of the object which caused low IoU scores.

Further Enhancements:
- **Larger Dataset**: Training on a larger dataset would improve the model's ability to generalize and produce more accurate code.
- **Clip Training**: Fine-tuning the CLIP model on the dataset could enhance the model's understanding of visual features relevant to CadQuery code generation.
- **Training Monitor Metrics**: Implementing metrics to monitor training progress and prevent overfitting would be beneficial.
- **DPO Training**: By applying Direct Preference Optimization (DPO) to the model to give a reward to the model for generating code which produce shapes similar to the target shape (high IoU), the model can be improved to generate code that is more aligned with the target shape.


Evaluation Logs:
Fine-tuned Model Evaluation
==================================================
Loading fine-tuned model...
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.33s/it]
Fine-tuned model loaded successfully
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
You have video processor config saved in `preprocessor.json` file which is deprecated. Video processor configs should be saved in their own `video_preprocessor.json` file. You can rename the file or load and save the processor back which renames it automatically. Loading from `preprocessor.json` will be removed in v5.0.
Loading test dataset (20 samples)...
Test dataset loaded
Running inference...
Sample 1/20 completed
Sample 2/20 completed
Sample 3/20 completed
Sample 4/20 completed
Sample 5/20 completed
Sample 6/20 completed
Sample 7/20 completed
Sample 8/20 completed
Sample 9/20 completed
Sample 10/20 completed
Sample 11/20 completed
Sample 12/20 completed
Sample 13/20 completed
Sample 14/20 completed
Sample 15/20 completed
Sample 16/20 completed
Sample 17/20 completed
Sample 18/20 completed
Sample 19/20 completed
Sample 20/20 completed

============================================================
EVALUATION RESULTS
============================================================

Valid Syntax Rate:
✓ 0: Successfully executed
✓ 1: Successfully executed
✓ 10: Successfully executed
✓ 11: Successfully executed
✓ 12: Successfully executed
✓ 13: Successfully executed
✓ 14: Successfully executed
✓ 15: Successfully executed
✓ 16: Successfully executed
✓ 17: Successfully executed
✓ 18: Successfully executed
✓ 19: Successfully executed
✓ 2: Successfully executed
✓ 3: Successfully executed
✓ 4: Successfully executed
✓ 5: Successfully executed
✓ 6: Successfully executed
✓ 7: Successfully executed
✓ 8: Successfully executed
✓ 9: Successfully executed

--- SUMMARY ---
Successful: 20/20
Valid Syntax Rate: 1.000
   VSR: 1.000 (100.0%)

IoU Evaluation:
Valid Syntax Rate: 1.000
Mean IOU_best   : 0.052
   Mean IoU Best: 0.052

Individual Sample Analysis:
----------------------------------------
Successful predictions: 20
Failed predictions: 0

Top 3 predictions by IoU:
   Sample 12: IoU = 0.281
   Sample 10: IoU = 0.146
   Sample 7: IoU = 0.086

Example Predictions:

--- Sample 0 ---
Generated Code:
  import cadquery as cq
  # Generating a workplane for sketch 0
  wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.75, -0.75, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
  loop0=wp_sketch0.moveTo(1.5, 0.0).lineTo(1.5, 1.5).lineTo(0.0, 1.5).lineTo(0.0, 0.0).close()
  solid0=wp_sketch0.add(loop0).extrude(0.0625)
  ... (11 more lines)

--- Sample 1 ---
Generated Code:
  import cadquery as cq
  # Generating a workplane for sketch 0
  wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.75, -0.296875, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
  loop0=wp_sketch0.moveTo(0.3421052631578947, 0.0).lineTo(0.3421052631578947, 0.5921052631578947).lineTo(0.0, 0.5921052631578947).lineTo(0.0, 0.0).close()
  solid0=wp_sketch0.add(loop0).extrude(0.078125)
  ... (6 more lines)

============================================================
FINAL SUMMARY
============================================================
Samples Evaluated: 20
Valid Syntax Rate: 1.000 (100.0%)
Mean IoU Best: 0.052
Successful Samples: 20/20
============================================================