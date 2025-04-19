#!/bin/bash

# First log in to Hugging Face with the token
if [ ! -z "$HF_TOKEN" ]; then
  echo "Logging into Hugging Face Hub..."
  huggingface-cli login --token $HF_TOKEN
fi
echo "TRYING TO RUN"
# Create virtual environment for packages
uv venv

# Install setuptools first - essential for builds
echo "Installing setuptools..."
uv pip install setuptools wheel

# Install torch with uv pip install
echo "Installing torch with uv pip install..."
uv pip install torch

# Install flash-attn dependencies
echo "Installing flash-attn build dependencies..."
uv pip install ninja packaging



# Install flash-attn with uv pip install and no build isolation
echo "Installing flash-attn with uv pip install..."
uv pip install flash-attn --no-build-isolation
# --- Configuration ---
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-VL-3B-Instruct"}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-144}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-8}
NUM_DEVICES=${NUM_DEVICES:-8} # Adjust if your hardware setup changed
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
OUTPUT_DIR=${OUTPUT_DIR:-"output/sophris_table_v2"} # Unified output directory for all runs
MERGED_MODEL_PATH=${MERGED_MODEL_PATH:-"${OUTPUT_DIR}_merged"} # Merged path based on output dir
DATA_DIR=${DATA_DIR:-"data/sophris-datasheet-table-extraction-azure-distill-v1"}
CONVERTED_DATA_DIR=${CONVERTED_DATA_DIR:-"data/llava-format"}
IMAGE_DIR=${IMAGE_DIR:-"data/images"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$OUTPUT_DIR/runs"} # Tensorboard logs
HUB_MODEL_ID=${HUB_MODEL_ID:-"ChunkrAI/sophris-table-VLM"}
VISUALIZATION_DIR=${VISUALIZATION_DIR:-"$OUTPUT_DIR/visualizations"}
NUM_EPOCHS=${NUM_EPOCHS:-100} # Total desired epochs (adjust if resuming)
DATA_LIMIT=${DATA_LIMIT:-48}

# --- Dynamic Checkpoint Detection ---
RESUME_CHECKPOINT=""
LATEST_CHECKPOINT=$(ls -td "$OUTPUT_DIR"/checkpoint-*/ 2>/dev/null | head -n 1)

if [ -d "$LATEST_CHECKPOINT" ]; then
    echo "Latest checkpoint found: $LATEST_CHECKPOINT"
    RESUME_CHECKPOINT=$LATEST_CHECKPOINT
    # Optional: Adjust NUM_EPOCHS if resuming, e.g., based on previous state
    # Or simply train for NUM_EPOCHS *additional* epochs from the checkpoint
else
    echo "No checkpoint found in $OUTPUT_DIR. Starting a new training run."
fi
# --- End Dynamic Checkpoint Detection ---

# Create required directories
mkdir -p $OUTPUT_DIR $CONVERTED_DATA_DIR $IMAGE_DIR $TENSORBOARD_DIR $VISUALIZATION_DIR $MERGED_MODEL_PATH

# Check if the resume checkpoint exists (only relevant if we found one)
# This check is now less critical as we dynamically find it, but kept for safety
if [ -n "$RESUME_CHECKPOINT" ] && [ ! -d "$RESUME_CHECKPOINT" ]; then
  echo "Error: Detected resume checkpoint directory does not exist: $RESUME_CHECKPOINT"
  echo "There might be an issue with checkpoint detection or directory structure."
  exit 1
fi

# Start TensorBoard with remote access enabled
echo "Starting TensorBoard in background..."
nohup tensorboard --logdir $TENSORBOARD_DIR --port 6006 --bind_all > tensorboard.log 2>&1 &
echo "TensorBoard started in background. Access at http://<your-sf-compute-ip>:6006"

# # Step 1: Prepare Dataset
# echo "Preparing dataset..."
# uv run prepare_dataset.py \
#     --output_dir $DATA_DIR \
#     --data_limit $DATA_LIMIT \
#     --train_ratio 0.8 \
#     --val_ratio 0.1 \
#     --test_ratio 0.1

# # Get the actual subdirectory where files are stored
# DATASET_BASENAME=$(basename "$DATA_DIR")
# PREPARED_DATA_DIR="$DATA_DIR/$DATASET_BASENAME"
# echo "Using dataset files from: $PREPARED_DATA_DIR"

# # Define the image directory (assuming images are within the prepared data dir)
# IMAGE_DIR="$PREPARED_DATA_DIR" # Or adjust if images are in a specific subfolder like "$PREPARED_DATA_DIR/images"

# # Step 2: Convert Dataset
# echo "Converting dataset to LLaVA format..."
# uv run convert_dataset.py \
#     --input_dir "$PREPARED_DATA_DIR" \
#     --output_dir $CONVERTED_DATA_DIR \
#     --image_dir $IMAGE_DIR \
#     --create_tensorboard_dir

# Step 3: Train Model - Start or Continue Training
TRAIN_ARGS=(
    "--output_dir" "$OUTPUT_DIR"
    "--overwrite_output_dir"
    "--num_train_epochs" "$NUM_EPOCHS"
    "--use_liger" "${USE_LIGER:-True}"
    "--lora_enable" "${LORA_ENABLE:-True}"
    "--use_dora" "${USE_DORA:-False}"
    "--lora_namespan_exclude" "${LORA_NAMESPAN_EXCLUDE:-"[]"}"
    "--lora_rank" "${LORA_RANK:-64}"
    "--lora_alpha" "${LORA_ALPHA:-64}"
    "--lora_dropout" "${LORA_DROPOUT:-0.05}"
    "--num_lora_modules" "${NUM_LORA_MODULES:--1}"
    "--deepspeed" "scripts/zero3_offload.json"
    "--model_id" "$MODEL_NAME"
    "--data_path" "$CONVERTED_DATA_DIR/train.json"
    "--image_folder" "$IMAGE_DIR"
    "--remove_unused_columns" "False"
    "--freeze_vision_tower" "${FREEZE_VISION_TOWER:-False}"
    "--freeze_llm" "${FREEZE_LLM:-True}"
    "--bf16" "${BF16:-True}"
    "--fp16" "${FP16:-False}"
    "--disable_flash_attn2" "${DISABLE_FLASH_ATTN2:-False}"
    "--per_device_train_batch_size" "$BATCH_PER_DEVICE"
    "--gradient_accumulation_steps" "$GRAD_ACCUM_STEPS"
    "--image_min_pixels" "$((256 * 28 * 28))"
    "--image_max_pixels" "$((1280 * 28 * 28))"
    "--learning_rate" "${LEARNING_RATE:-1e-4}"
    "--merger_lr" "${MERGER_LR:-1e-5}"
    "--vision_lr" "${VISION_LR:-5e-6}"
    "--weight_decay" "${WEIGHT_DECAY:-0.1}"
    "--warmup_ratio" "${WARMUP_RATIO:-0.03}"
    "--lr_scheduler_type" "${LR_SCHEDULER_TYPE:-cosine}"
    "--logging_steps" "${LOGGING_STEPS:-1}"
    "--tf32" "${TF32:-True}"
    "--gradient_checkpointing" "${GRADIENT_CHECKPOINTING:-True}"
    "--report_to" "tensorboard"
    "--lazy_preprocess" "${LAZY_PREPROCESS:-True}"
    "--save_strategy" "${SAVE_STRATEGY:-steps}"
    "--save_steps" "${SAVE_STEPS:-200}"
    "--save_total_limit" "${SAVE_TOTAL_LIMIT:-10}"
    "--dataloader_num_workers" "${DATALOADER_NUM_WORKERS:-4}"
    "--push_to_hub" "${PUSH_TO_HUB:-True}"
    "--hub_model_id" "$HUB_MODEL_ID"
    "--hub_private_repo" "${HUB_PRIVATE_REPO:-True}"
    "--hub_strategy" "${HUB_STRATEGY:-checkpoint}"
)

if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming training from checkpoint: $RESUME_CHECKPOINT"
    echo "Saving new checkpoints and logs to: $OUTPUT_DIR"
    TRAIN_ARGS+=("--resume_from_checkpoint" "$RESUME_CHECKPOINT")
else
    echo "Starting new training run."
    echo "Saving checkpoints and logs to: $OUTPUT_DIR"
fi

# Ensure PYTHONPATH is set for deepspeed launch
uv run deepspeed src/training/train.py "${TRAIN_ARGS[@]}"

# Step 4: Merge LoRA Weights from the *final* state in OUTPUT_DIR
echo "Merging LoRA weights from the run..."
uv run src/merge_lora_weights.py \
    --model-path $OUTPUT_DIR \
    --model-base $MODEL_NAME \
    --save-model-path $MERGED_MODEL_PATH \
    --hub-model-id $HUB_MODEL_ID # Optionally change commit message here if needed

# Step 5: Evaluate Model using the merged model from OUTPUT_DIR
echo "Evaluating the merged model..."
uv run src/training/evaluation.py \
    --model_path $MERGED_MODEL_PATH \
    --data_path $CONVERTED_DATA_DIR/test.json \
    --image_folder $IMAGE_DIR \
    --output_dir $OUTPUT_DIR/evaluation_results \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --use_flash_attn True \
    --device cuda \
    --hub_model_id $HUB_MODEL_ID # Optionally change commit message

# Step 6: Visualize Results (Commented out old static version)
# echo "Visualizing results..."
# uv run src/training/visualize_results.py \
#     --results_path $OUTPUT_DIR/evaluation_results/eval_scores.json \
#     --output_dir $VISUALIZATION_DIR \
#     --image_dir $IMAGE_DIR \
#     --tensorboard_dir $TENSORBOARD_DIR \
#     --hub_model_id $HUB_MODEL_ID # Optionally change commit message

echo "Pipeline complete. Checkpoints, logs, and results pushed to $HUB_MODEL_ID (if enabled)"

# Remind about the tensorboard session
if command -v tmux &> /dev/null; then
    echo ""
    echo "TensorBoard is still running in tmux session 'tensorboard_sophris'"
    echo "- View at http://localhost:6006"
    echo "- Attach: tmux attach -t tensorboard_sophris"
    echo "- Kill: tmux kill-session -t tensorboard_sophris"
fi

# Step 7: Launch Interactive Evaluation Viewer (Optional)
echo ""
echo "Launching interactive evaluation viewer..."
echo "Access it at http://localhost:7860 (or the configured port)"
# Note: This will run in the foreground. You might want to run it in the background
# or in a separate terminal/tmux session for long-running access.
# The image directory finding is heuristic; adjust pattern if needed.
uv run src/app/evaluation_app.py \
    --base_output_dir "$(dirname "$OUTPUT_DIR")" \
    --initial_run_dir "$OUTPUT_DIR" \
    --image_base_dir_pattern "$DATA_DIR/*" # Adjust pattern based on where images *actually* are relative to DATA_DIR 