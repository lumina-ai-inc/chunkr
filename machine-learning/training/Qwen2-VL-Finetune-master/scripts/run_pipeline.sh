#!/bin/bash

# Configuration
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
OUTPUT_DIR="output/sophris_lora"
MERGED_MODEL_PATH="${OUTPUT_DIR}_merged"
DATA_DIR="data/tables-vlm-azure-distill-v1"
CONVERTED_DATA_DIR="data/llava-format"
IMAGE_DIR="data/images"
TENSORBOARD_DIR="$OUTPUT_DIR/runs"

# Create required directories
mkdir -p $OUTPUT_DIR $CONVERTED_DATA_DIR $IMAGE_DIR

# Launch TensorBoard in a tmux session
if command -v tmux &> /dev/null; then
    echo "Starting TensorBoard in a tmux session..."
    # Kill existing session if it exists
    tmux kill-session -t tensorboard 2>/dev/null || true
    
    # Create a new detached session
    tmux new-session -d -s tensorboard "tensorboard --logdir $TENSORBOARD_DIR --bind_all && bash"
    
    echo "TensorBoard started in tmux session. To view:"
    echo "  1. Access http://localhost:6006 in your browser"
    echo "  2. Or attach to the tmux session: tmux attach -t tensorboard"
else
    echo "tmux not found. Install tmux to automatically launch TensorBoard."
    echo "You can manually run: tensorboard --logdir $TENSORBOARD_DIR"
fi

# Step 1: Prepare Dataset
echo "Preparing dataset..."
uv run prepare_dataset.py \
    --output_dir $DATA_DIR \
    --data_limit 2000 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1

# Get the actual subdirectory where files are stored
DATASET_BASENAME=$(basename "$DATA_DIR")
PREPARED_DATA_DIR="$DATA_DIR/$DATASET_BASENAME"
echo "Using dataset files from: $PREPARED_DATA_DIR"

# Step 2: Convert Dataset
echo "Converting dataset to LLaVA format..."
uv run convert_dataset.py \
    --input_dir "$PREPARED_DATA_DIR" \
    --output_dir $CONVERTED_DATA_DIR \
    --image_dir $IMAGE_DIR \
    --create_tensorboard_dir

# Step 3: Train Model with LoRA
echo "Training model with LoRA..."
# Ensure PYTHONPATH is set for deepspeed launch
export PYTHONPATH=src:$PYTHONPATH
uv run deepspeed src/training/train.py \
    --use_liger True \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path $CONVERTED_DATA_DIR/train.json \
    --image_folder $IMAGE_DIR \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 10 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-4 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4

# Step 4: Merge LoRA Weights
echo "Merging LoRA weights..."
uv run bash scripts/merge_lora.sh $MODEL_NAME $OUTPUT_DIR $MERGED_MODEL_PATH

# Step 5: Evaluate Model
echo "Evaluating model..."
uv run src/training/evaluation.py \
    --model_path $MERGED_MODEL_PATH \
    --data_path $CONVERTED_DATA_DIR/test.json \
    --image_folder $IMAGE_DIR \
    --output_dir $TENSORBOARD_DIR \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --use_flash_attn True \
    --device cuda

echo "Pipeline complete. Results saved to $TENSORBOARD_DIR"

# Remind about the tensorboard session
if command -v tmux &> /dev/null; then
    echo ""
    echo "TensorBoard is still running in tmux session 'tensorboard'"
    echo "- View at http://localhost:6006"
    echo "- Attach to session: tmux attach -t tensorboard"
    echo "- Kill session when done: tmux kill-session -t tensorboard"
fi 