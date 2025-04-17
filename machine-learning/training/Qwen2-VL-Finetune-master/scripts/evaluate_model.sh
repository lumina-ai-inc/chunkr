#!/bin/bash

MODEL_PATH="output/my_lora_model"  # Path to trained model
TEST_DATA="data/llava-format/test.json"  # Path to test data
IMAGE_DIR="data/images"  # Path to images directory
OUTPUT_DIR="evaluation_results"  # Where to save results

export PYTHONPATH=src:$PYTHONPATH

mkdir -p $OUTPUT_DIR

# Run evaluation on test set
python src/training/evaluation.py \
    --model_path $MODEL_PATH \
    --data_path $TEST_DATA \
    --image_folder $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --use_flash_attn True \
    --device cuda

# Generate visualization of results
python src/utils/visualize_results.py \
    --results_path $OUTPUT_DIR/results.json \
    --output_dir $OUTPUT_DIR/visualizations

echo "Evaluation complete. Results saved to $OUTPUT_DIR" 