#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

DATA_DIR="data/tables-vlm-azure-distill-v1"
CONVERTED_DATA_DIR="data/llava-format"
IMAGE_DIR="data/images"
EVAL_OUTPUT_DIR="evaluation_results"

# Step 1: Prepare Dataset
echo "Preparing dataset..."
python prepare_dataset.py \
    --output_dir $DATA_DIR \
    --data_limit 1000 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1

# Construct the path where prepare_dataset actually saves the files
# It creates a subdirectory named after the basename of DATA_DIR
DATASET_BASENAME=$(basename "$DATA_DIR")
PREPARED_DATA_DIR="$DATA_DIR/$DATASET_BASENAME"
echo "Constructed prepared data path: $PREPARED_DATA_DIR"

# Step 2: Convert Dataset
echo "Converting dataset to LLaVA format..."
python convert_dataset.py \
    --input_dir "$PREPARED_DATA_DIR" \
    --output_dir $CONVERTED_DATA_DIR \
    --image_dir $IMAGE_DIR \
    --create_tensorboard_dir

# Step 3: Train Model with LoRA
echo "Training model with LoRA..."
export PYTHONPATH=src:$PYTHONPATH
deepspeed src/training/train.py \
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
    --data_path data/llava-format/train.json \
    --image_folder data/images \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/sophris_lora \
    --num_train_epochs 1 \
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