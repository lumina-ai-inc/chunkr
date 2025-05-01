#!/usr/bin/env bash
# train_qwen2vl_7b.sh  —  LoRA finetune with DeepSpeed-Zero-3
set -euo pipefail

# ---------------- Authentication ----------------
[[ -n "${HF_TOKEN:-}" ]] && huggingface-cli login --token "${HF_TOKEN}"

# ---------------- Virtual-env & deps ----------------
uv venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install torch ninja packaging flash-attn --no-build-isolation
uv pip install transformers datasets accelerate deepspeed tensorboard

# ---------------- Hyper-parameters ----------------
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-VL-7B-Instruct"}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-2}
NUM_DEVICES=${NUM_DEVICES:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}          # ← keep JSON in sync
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE*NUM_DEVICES)))

OUTPUT_DIR=${OUTPUT_DIR:-"output/sophris_table_vllm_compat_7b"}
MERGED_MODEL_PATH=${MERGED_MODEL_PATH:-"${OUTPUT_DIR}_merged"}
DATA_DIR=${DATA_DIR:-"data/sophris-datasheet-table-extraction-azure-distill-v1"}
CONVERTED_DATA_DIR=${CONVERTED_DATA_DIR:-"data/llava-format"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$OUTPUT_DIR/runs"}
CHECKPOINT_HUB_MODEL_ID=${HUB_MODEL_ID:-"ChunkrAI/chunkr-table-v1-qwen2_5VL-7B-LORA"}
HUB_MODEL_ID=${HUB_MODEL_ID:-"ChunkrAI/chunkr-table-v1-qwen2_5VL-7B"}
NUM_EPOCHS=${NUM_EPOCHS:-10}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# ---------------- Auto-resume ----------------
RESUME_CHECKPOINT=""
if LATEST=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -n1); then
  RESUME_CHECKPOINT=$LATEST
  echo "[+] Resuming from $LATEST"
fi

# ---------------- Folders ----------------
mkdir -p "$OUTPUT_DIR" "$CONVERTED_DATA_DIR" "$TENSORBOARD_DIR" "$MERGED_MODEL_PATH"

# ---------------- TensorBoard ----------------
nohup tensorboard --logdir "$TENSORBOARD_DIR" --port 6006 --bind_all \
      > tensorboard.log 2>&1 &

# =================================================
# Step 1: Prepare Dataset
echo "Preparing dataset..."
# uv run prepare_dataset.py \
#     --output_dir $DATA_DIR \
#     --data_limit $DATA_LIMIT \
#     --train_ratio 0.8 \
#     --val_ratio 0.1 \
#     --test_ratio 0.1
#
# Get the actual subdirectory where files are stored
DATASET_BASENAME=$(basename "$DATA_DIR")
PREPARED_DATA_DIR="$DATA_DIR/$DATASET_BASENAME"
echo "Using dataset files from: $PREPARED_DATA_DIR"

# Define the image directory (prefers sub-folder if present)
if [[ -d "$PREPARED_DATA_DIR/images" ]]; then
  IMAGE_DIR="$PREPARED_DATA_DIR/images"
else
  IMAGE_DIR="$PREPARED_DATA_DIR"
fi
echo "Images located at: $IMAGE_DIR"

# Step 2: Convert Dataset
echo "Converting dataset to LLaVA format..."
# uv run convert_dataset.py \
#     --input_dir "$PREPARED_DATA_DIR" \
#     --output_dir $CONVERTED_DATA_DIR \
#     --image_dir $IMAGE_DIR \
#     --create_tensorboard_dir

# Step 3: Train Model - Start or Continue Training
# =================================================

# ---------------- DeepSpeed config sync ----------------
DS_JSON="scripts/zero3.json"
sed -i -E \
  -e "s/\"train_micro_batch_size_per_gpu\": *[0-9]+/\"train_micro_batch_size_per_gpu\": $BATCH_PER_DEVICE/" \
  -e "s/\"gradient_accumulation_steps\": *[0-9]+/\"gradient_accumulation_steps\": $GRAD_ACCUM_STEPS/" \
  -e "s/\"train_batch_size\": *[0-9]+/\"train_batch_size\": $GLOBAL_BATCH_SIZE/" \
  "$DS_JSON"

TRAIN_ARGS=(
  --output_dir "$OUTPUT_DIR" --overwrite_output_dir
  --num_train_epochs "$NUM_EPOCHS"
  --use_liger True
  --lora_enable True
  --lora_namespan_exclude "['lm_head','embed_tokens']"
  --lora_rank 64 --lora_alpha 64 --lora_dropout 0.05
  --deepspeed "$DS_JSON"
  --model_id "$MODEL_NAME"
  --data_path "$CONVERTED_DATA_DIR/train.json"
  --image_folder "$IMAGE_DIR"
  --remove_unused_columns False
  --freeze_vision_tower True --freeze_llm True
  --bf16 True
  --per_device_train_batch_size "$BATCH_PER_DEVICE"
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
  --max_grad_norm "$MAX_GRAD_NORM"
  --group_by_length True
  --learning_rate 1e-7 --merger_lr 1e-7 --vision_lr 1e-7
  --weight_decay 0.1 --warmup_ratio 0.1
  --lr_scheduler_type cosine --logging_steps 1
  --tf32 True --gradient_checkpointing True
  --report_to tensorboard --lazy_preprocess True
  --save_strategy steps --save_steps 200 --save_total_limit 10
  --dataloader_num_workers 2
  --push_to_hub True --hub_model_id "$CHECKPOINT_HUB_MODEL_ID" \
  --hub_private_repo True --hub_strategy checkpoint
  --max_seq_length 1024
  --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5
)

[[ -n "$RESUME_CHECKPOINT" ]] && \
  TRAIN_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")

echo "[+] Launching DeepSpeed…"
uv run deepspeed src/training/train.py "${TRAIN_ARGS[@]}"

# ---------------- Merge LoRA ----------------
uv run src/merge_lora_weights.py \
  --model-path "$OUTPUT_DIR" --model-base "$MODEL_NAME" \
  --save-model-path "$MERGED_MODEL_PATH" --hub-model-id "$HUB_MODEL_ID"

# ---------------- Evaluate ----------------
uv run src/training/evaluation.py \
  --model_path "$MERGED_MODEL_PATH" \
  --data_path "$CONVERTED_DATA_DIR/test.json" \
  --image_folder "$IMAGE_DIR" \
  --output_dir "$OUTPUT_DIR/evaluation_results" \
  --batch_size 4 --max_new_tokens 1024 \
  --use_flash_attn True --device cuda \
  --hub_model_id "$HUB_MODEL_ID"

echo "[✓] Training, merge, and eval complete."
