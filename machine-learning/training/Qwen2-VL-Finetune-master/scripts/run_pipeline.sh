#!/bin/bash

# --- Configuration ---
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
GLOBAL_BATCH_SIZE=144
BATCH_PER_DEVICE=6
NUM_DEVICES=8 # Adjust if your hardware setup changed
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
OUTPUT_DIR="output/sophris_table_v1" # Unified output directory for all runs
MERGED_MODEL_PATH="${OUTPUT_DIR}_merged" # Merged path based on output dir
DATA_DIR="data/sophris-datasheet-table-extraction-azure-distill-v1"
CONVERTED_DATA_DIR="data/llava-format"
IMAGE_DIR="data/images"
TENSORBOARD_DIR="$OUTPUT_DIR/runs" # Tensorboard logs
HUB_MODEL_ID="ChunkrAI/sophris-table-VLM"
VISUALIZATION_DIR="$OUTPUT_DIR/visualizations"
NUM_EPOCHS=100 # Total desired epochs (adjust if resuming)

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

# Ensure user is logged in to Hugging Face Hub
if ! huggingface-cli whoami > /dev/null 2>&1; then
  echo "You are not logged into Hugging Face Hub. Please run 'huggingface-cli login'"
  # Optionally exit if login is mandatory: exit 1
  # Or prompt for login: huggingface-cli login
fi

# Create required directories
mkdir -p $OUTPUT_DIR $CONVERTED_DATA_DIR $IMAGE_DIR $TENSORBOARD_DIR $VISUALIZATION_DIR $MERGED_MODEL_PATH

# Check if the resume checkpoint exists (only relevant if we found one)
# This check is now less critical as we dynamically find it, but kept for safety
# if [ -n "$RESUME_CHECKPOINT" ] && [ ! -d "$RESUME_CHECKPOINT" ]; then
#   echo "Error: Detected resume checkpoint directory does not exist: $RESUME_CHECKPOINT"
#   echo "There might be an issue with checkpoint detection or directory structure."
#   exit 1
# fi

# Launch TensorBoard
# First, ensure any lingering TensorBoard processes are stopped
echo "Attempting to stop any existing TensorBoard processes..."
pkill -f tensorboard # Forcefully kill processes matching 'tensorboard'
sleep 2 # Give processes a moment to terminate

if command -v tmux &> /dev/null; then
    echo "Starting TensorBoard in a tmux session..."
    tmux kill-session -t tensorboard_sophris 2>/dev/null || true # Kill the specific tmux session if it exists
    echo "Launching TensorBoard with logdir: $TENSORBOARD_DIR" # Verify the path being used
    tmux new-session -d -s tensorboard_sophris "tensorboard --logdir $TENSORBOARD_DIR --bind_all --port 6006 && bash" # Use a consistent port
    echo "TensorBoard started in tmux session 'tensorboard_sophris'. To view:"
    echo "  1. Access http://localhost:6006 in your browser"
    echo "  2. Or attach: tmux attach -t tensorboard_sophris"
else
    echo "tmux not found. Install tmux to automatically launch TensorBoard."
    echo "You can manually run: tensorboard --logdir $TENSORBOARD_DIR --port 6006"
fi

# Step 1: Prepare Dataset
echo "Preparing dataset..."
uv run prepare_dataset.py \
    --output_dir $DATA_DIR \
    --data_limit 24000 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1

# Get the actual subdirectory where files are stored
DATASET_BASENAME=$(basename "$DATA_DIR")
PREPARED_DATA_DIR="$DATA_DIR/$DATASET_BASENAME"
echo "Using dataset files from: $PREPARED_DATA_DIR"

# Define the image directory (assuming images are within the prepared data dir)
IMAGE_DIR="$PREPARED_DATA_DIR" # Or adjust if images are in a specific subfolder like "$PREPARED_DATA_DIR/images"

# Step 2: Convert Dataset
echo "Converting dataset to LLaVA format..."
uv run convert_dataset.py \
    --input_dir "$PREPARED_DATA_DIR" \
    --output_dir $CONVERTED_DATA_DIR \
    --image_dir $IMAGE_DIR \
    --create_tensorboard_dir

# Step 3: Train Model - Start or Continue Training
TRAIN_ARGS=(
    "--output_dir" "$OUTPUT_DIR"
    "--overwrite_output_dir" # Overwrite necessary for resuming/consistent output
    "--num_train_epochs" "$NUM_EPOCHS"
    "--use_liger" "True"
    "--lora_enable" "True"
    "--use_dora" "False"
    "--lora_namespan_exclude" "['lm_head', 'embed_tokens']"
    "--lora_rank" "64"
    "--lora_alpha" "64"
    "--lora_dropout" "0.05"
    "--num_lora_modules" "-1"
    "--deepspeed" "scripts/zero3_offload.json"
    "--model_id" "$MODEL_NAME"
    "--data_path" "$CONVERTED_DATA_DIR/train.json"
    "--image_folder" "$IMAGE_DIR"
    "--remove_unused_columns" "False"
    "--freeze_vision_tower" "False"
    "--freeze_llm" "True"
    "--bf16" "True"
    "--fp16" "False"
    "--disable_flash_attn2" "False"
    "--per_device_train_batch_size" "$BATCH_PER_DEVICE"
    "--gradient_accumulation_steps" "$GRAD_ACCUM_STEPS"
    "--image_min_pixels" "$((256 * 28 * 28))"
    "--image_max_pixels" "$((1280 * 28 * 28))"
    "--learning_rate" "1e-4"
    "--merger_lr" "1e-5"
    "--vision_lr" "2e-6"
    "--weight_decay" "0.1"
    "--warmup_ratio" "0.03"
    "--lr_scheduler_type" "cosine"
    "--logging_steps" "1"
    "--tf32" "True"
    "--gradient_checkpointing" "True"
    "--report_to" "tensorboard"
    "--lazy_preprocess" "True"
    "--save_strategy" "steps"
    "--save_steps" "200"
    "--save_total_limit" "10"
    "--dataloader_num_workers" "4"
    "--push_to_hub" "True"
    "--hub_model_id" "$HUB_MODEL_ID"
    "--hub_private_repo" "True"
    "--hub_strategy" "checkpoint"
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
export PYTHONPATH=src:$PYTHONPATH
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