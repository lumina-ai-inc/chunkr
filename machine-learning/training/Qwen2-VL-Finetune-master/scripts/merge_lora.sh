#!/bin/bash

# Use variables that match run_pipeline.sh format
MODEL_NAME=${1:-"Qwen/Qwen2.5-VL-3B-Instruct"}
LORA_MODEL_PATH=${2:-"output/sophris_lora"}
MERGED_MODEL_PATH=${3:-"${LORA_MODEL_PATH}_merged"}

# Create output directory
mkdir -p $MERGED_MODEL_PATH

# Set Python path
echo "Merging LoRA weights..."
echo "Base model: $MODEL_NAME"
echo "LoRA model: $LORA_MODEL_PATH"
echo "Output path: $MERGED_MODEL_PATH"

python src/merge_lora_weights.py \
    --model-path $LORA_MODEL_PATH \
    --model-base $MODEL_NAME \
    --save-model-path $MERGED_MODEL_PATH \
    --safe-serialization

echo "LoRA weights merged successfully to $MERGED_MODEL_PATH"