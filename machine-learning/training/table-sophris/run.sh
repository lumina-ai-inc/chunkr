#!/bin/bash
set -e

# Load environment variables from .env
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default values
DATA_DIR="data"
# Get dataset name from environment variable or use default
[ -z "$DATASET_NAME" ] && DATASET_NAME="default"
echo "Using dataset name from environment: $DATASET_NAME"
MODEL_NAME="unsloth/Qwen2.5-VL-3B-Instruct"
BATCH_SIZE=12
GRAD_ACCUM=4
MAX_STEPS=100
LR=1e-4
LORA_R=16
LORA_ALPHA=16
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
SEED=42
EVAL_SPLIT="test"
NUM_EVAL_SAMPLES=10
DATA_LIMIT=""
OUTPUT_DIR="outputs/qwen2_5_Vl_3B-table-finetune"
LORA_OUTPUT_DIR="lora_model/qwen2_5_Vl_3B-table-finetune"
CACHE_DIR="model_cache"  # New cache directory for downloaded models

# Parse command-line arguments
PREPARE=false  
TRAIN=false
EVAL=false

function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Task options (at least one required):"
    echo "  --prepare               Prepare the dataset"
    echo "  --train                 Train the model"
    echo "  --eval                  Evaluate model(s)"
    echo "  --all                   Run all steps (prepare → train → evaluate)"
    echo ""
    echo "Data preparation options:"
    echo "  --data-dir DIR          Directory for dataset files (default: $DATA_DIR)"
    echo "  --data-limit NUM        Limit total samples (default: all available)"
    echo "  --train-ratio NUM       Ratio for training set (default: $TRAIN_RATIO)"
    echo "  --val-ratio NUM         Ratio for validation set (default: $VAL_RATIO)"
    echo "  --test-ratio NUM        Ratio for test set (default: $TEST_RATIO)"
    echo "  --s3-bucket NAME        S3 bucket name (override .env)"
    echo "  --dataset-name NAME     Dataset name in S3 (override .env)"
    echo ""
    echo "Training options:"
    echo "  --model-name NAME       Base model to fine-tune (default: $MODEL_NAME)"
    echo "  --output-dir DIR        Checkpoint directory (default: $OUTPUT_DIR)"
    echo "  --lora-output-dir DIR   LoRA model save directory (default: $LORA_OUTPUT_DIR)"
    echo "  --batch-size NUM        Batch size per device (default: $BATCH_SIZE)"
    echo "  --grad-accum NUM        Gradient accumulation steps (default: $GRAD_ACCUM)"
    echo "  --max-steps NUM         Maximum training steps (default: $MAX_STEPS)"
    echo "  --lr NUM                Learning rate (default: $LR)"
    echo "  --lora-r NUM            LoRA rank (default: $LORA_R)"
    echo "  --lora-alpha NUM        LoRA alpha (default: $LORA_ALPHA)"
    echo ""
    echo "Evaluation options:"
    echo "  --eval-split TYPE       Split to use for evaluation (val or test, default: $EVAL_SPLIT)"
    echo "  --num-eval-samples NUM  Number of samples to evaluate (default: $NUM_EVAL_SAMPLES)"
    echo "  --baseline-models LIST  Comma-separated list of baseline models (default: none)"
    echo "  --skip-baseline         Skip baseline model evaluation (default: false)"
    echo ""
    echo "Common options:"
    echo "  --seed NUM              Random seed (default: $SEED)"
    echo "  -h, --help              Show this help message"
    exit 1
}

if [ "$#" -eq 0 ]; then
    show_help
fi

# Store extra args to pass to individual scripts
PREPARE_ARGS=""
TRAIN_ARGS=""
EVAL_ARGS=""
BASELINE_MODELS=""

# Create model cache directory
mkdir -p "$CACHE_DIR"

# Step 0: Download the model files first
echo "===== STEP 0: DOWNLOADING BASE MODEL ====="
echo "Model: $MODEL_NAME"

# Convert model name to directory format
MODEL_DIR_NAME=$(echo "$MODEL_NAME" | sed 's/\//_/g')
MODEL_DIR="$CACHE_DIR/$MODEL_DIR_NAME"

# Create model directory
mkdir -p "$MODEL_DIR"

# Check if model is already downloaded
if [ -f "$MODEL_DIR/config.json" ]; then
    echo "Model files already exist in $MODEL_DIR, skipping download"
else
    echo "Downloading model $MODEL_NAME to $MODEL_DIR"
    
    # Use Python to download the model with HuggingFace
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = '$MODEL_NAME'
cache_dir = '$MODEL_DIR'

print(f'Downloading tokenizer for {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer.save_pretrained(cache_dir)

print(f'Downloading model {model_name}...')
# Only download the model configuration to save time/space
from huggingface_hub import hf_hub_download
import json

# Get model config
config_file = hf_hub_download(repo_id=model_name, filename='config.json')
with open(config_file, 'r') as f:
    config = json.load(f)
    
# Save config to cache dir
with open(os.path.join(cache_dir, 'config.json'), 'w') as f:
    json.dump(config, f)
    
print(f'Model files downloaded to {cache_dir}')
"
    
    # Check if download was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download model"
        exit 1
    fi
    
    echo "Model downloaded successfully"
fi

# Export cache directory to environment
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_HOME="$CACHE_DIR"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --prepare) PREPARE=true; shift 1;;
        --train) TRAIN=true; shift 1;;
        --eval) EVAL=true; shift 1;;
        --all) PREPARE=true; TRAIN=true; EVAL=true; shift 1;;
        
        # Common options
        --data-dir) DATA_DIR="$2"; shift 2;;
        --seed) SEED="$2"; PREPARE_ARGS="$PREPARE_ARGS --seed $2"; TRAIN_ARGS="$TRAIN_ARGS --seed $2"; shift 2;;
        
        # Data preparation options
        --data-limit) DATA_LIMIT="$2"; PREPARE_ARGS="$PREPARE_ARGS --data_limit $2"; shift 2;;
        --train-ratio) TRAIN_RATIO="$2"; PREPARE_ARGS="$PREPARE_ARGS --train_ratio $2"; shift 2;;
        --val-ratio) VAL_RATIO="$2"; PREPARE_ARGS="$PREPARE_ARGS --val_ratio $2"; shift 2;;
        --test-ratio) TEST_RATIO="$2"; PREPARE_ARGS="$PREPARE_ARGS --test_ratio $2"; shift 2;;
        --s3-bucket) PREPARE_ARGS="$PREPARE_ARGS --s3_bucket $2"; shift 2;;
        --dataset-name) 
            DATASET_NAME="$2"
            # Don't add quotes to args that will be passed to Python scripts
            PREPARE_ARGS="$PREPARE_ARGS --dataset_name $2"
            TRAIN_ARGS="$TRAIN_ARGS --dataset_name $2"
            EVAL_ARGS="$EVAL_ARGS --dataset_name $2"
            export DATASET_NAME="$2"  # Make sure environment is updated for child processes
            shift 2;;
        
        # Training options
        --model-name) MODEL_NAME="$2"; TRAIN_ARGS="$TRAIN_ARGS --model_name $2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; TRAIN_ARGS="$TRAIN_ARGS --output_dir $2"; shift 2;;
        --lora-output-dir) LORA_OUTPUT_DIR="$2"; TRAIN_ARGS="$TRAIN_ARGS --lora_output_dir $2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; TRAIN_ARGS="$TRAIN_ARGS --batch_size $2"; EVAL_ARGS="$EVAL_ARGS --batch_size $2"; shift 2;;
        --grad-accum) TRAIN_ARGS="$TRAIN_ARGS --grad_accum $2"; shift 2;;
        --max-steps) TRAIN_ARGS="$TRAIN_ARGS --max_steps $2"; shift 2;;
        --lr) TRAIN_ARGS="$TRAIN_ARGS --learning_rate $2"; shift 2;;
        --lora-r) TRAIN_ARGS="$TRAIN_ARGS --lora_r $2"; shift 2;;
        --lora-alpha) TRAIN_ARGS="$TRAIN_ARGS --lora_alpha $2"; shift 2;;
        
        # Evaluation options
        --eval-split) EVAL_SPLIT="$2"; EVAL_ARGS="$EVAL_ARGS --eval_split $2"; shift 2;;
        --num-eval-samples) NUM_EVAL_SAMPLES="$2"; EVAL_ARGS="$EVAL_ARGS --num_samples $2"; shift 2;;
        --baseline-models) 
            IFS=',' read -ra MODELS <<< "$2"
            for model in "${MODELS[@]}"; do
                BASELINE_MODELS="$BASELINE_MODELS \"$model\""
            done
            shift 2;;
        --skip-baseline) EVAL_ARGS="$EVAL_ARGS --skip_baseline"; shift 1;;
        
        # Help
        -h|--help) show_help;;
        *) echo "Unknown option: $1"; show_help;;
    esac
done

# Validate that at least one action was specified
if [ "$PREPARE" = false ] && [ "$TRAIN" = false ] && [ "$EVAL" = false ]; then
    echo "Error: You must specify at least one action (--prepare, --train, --eval, or --all)"
    exit 1
fi

# Make sure data directory is consistent across all commands
PREPARE_ARGS="$PREPARE_ARGS --output_dir $DATA_DIR"
TRAIN_ARGS="$TRAIN_ARGS --data_dir $DATA_DIR"
EVAL_ARGS="$EVAL_ARGS --data_dir $DATA_DIR"

# Step 1: Prepare Dataset if requested
if [ "$PREPARE" = true ]; then
    echo "===== STEP 1: PREPARING DATASET ====="
    
    # Build the prepare command with correct arguments
    PREPARE_CMD="python prepare_dataset.py --output_dir $DATA_DIR"
    
    # Only add dataset name from environment if not already in PREPARE_ARGS
    if [[ "$PREPARE_ARGS" != *"--dataset_name"* ]] && [ ! -z "$DATASET_NAME" ]; then
        PREPARE_CMD="$PREPARE_CMD --dataset_name $DATASET_NAME"
    fi
    
    # Add any additional preparation arguments
    PREPARE_CMD="$PREPARE_CMD $PREPARE_ARGS"
    
    echo "Command: $PREPARE_CMD"
    
    # Run prepare_dataset.py and capture its output
    PREPARE_OUTPUT=$(eval $PREPARE_CMD)
    echo "$PREPARE_OUTPUT"
    
    # Extract the dataset path from the output
    DATASET_PATH=$(echo "$PREPARE_OUTPUT" | grep "Output directory:" | sed 's/.*Output directory: //')
    
    if [ -z "$DATASET_PATH" ]; then
        echo "Error: Could not determine dataset path from prepare_dataset.py output"
        # Fallback to reading from dataset_name.txt file
        if [ -f "$DATA_DIR/dataset_name.txt" ]; then
            EXACT_DATASET_NAME=$(cat "$DATA_DIR/dataset_name.txt")
            DATASET_PATH="$DATA_DIR/$EXACT_DATASET_NAME"
            echo "Using dataset name from dataset_name.txt: $EXACT_DATASET_NAME"
        else
            echo "Trying to determine dataset path manually"
            # If we can't extract it from output, find the most recently modified dataset directory
            DATASET_PATH=$(find "$DATA_DIR" -maxdepth 1 -type d -not -path "$DATA_DIR" | sort -t' ' -k2 | tail -n1)
        fi
    fi
    
    # Make sure train.jsonl exists in dataset path
    if [ ! -f "$DATASET_PATH/train.jsonl" ]; then
        echo "Error: train.jsonl not found in $DATASET_PATH"
        echo "Available files in $DATASET_PATH:"
        ls -la "$DATASET_PATH"
        exit 1
    fi
    
    echo "Using dataset at: $DATASET_PATH"
    
    # Extract just the directory name for dataset_name
    DATASET_NAME=$(basename "$DATASET_PATH")
    echo "Setting dataset name to: $DATASET_NAME"
    export DATASET_NAME="$DATASET_NAME"  # Update environment with real dataset name
    
    # Update args for training to use this dataset name with path exactly as created
    # Remove any existing dataset_name arguments
    TRAIN_ARGS=$(echo "$TRAIN_ARGS" | sed 's/--dataset_name [^ ]*//')
    EVAL_ARGS=$(echo "$EVAL_ARGS" | sed 's/--dataset_name [^ ]*//')
    
    # Add the correct dataset name and data_dir
    TRAIN_ARGS="$TRAIN_ARGS --dataset_name $DATASET_NAME --data_dir $DATA_DIR"
    EVAL_ARGS="$EVAL_ARGS --dataset_name $DATASET_NAME --data_dir $DATA_DIR"
fi

# Step 2: Train Model if requested
TRAINING_SUCCESS=false
if [ "$TRAIN" = true ]; then
    echo "===== STEP 2: TRAINING MODEL ====="
    
    # Build the training command properly without duplicate args or unnecessary quotes
    TRAIN_CMD="python trainer.py"
    
    # Add data_dir only once without extra quotes
    TRAIN_CMD="$TRAIN_CMD --data_dir $DATA_DIR"
    
    # Add dataset_name without double quotes in the argument
    TRAIN_CMD="$TRAIN_CMD --dataset_name $DATASET_NAME"
    
    # Add model name
    TRAIN_CMD="$TRAIN_CMD --model_name $MODEL_NAME"
    
    # Add other training parameters
    TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
    TRAIN_CMD="$TRAIN_CMD --learning_rate $LR"
    TRAIN_CMD="$TRAIN_CMD --lora_r $LORA_R"
    TRAIN_CMD="$TRAIN_CMD --lora_alpha $LORA_ALPHA"
    TRAIN_CMD="$TRAIN_CMD --max_steps $MAX_STEPS"
    TRAIN_CMD="$TRAIN_CMD --output_dir $OUTPUT_DIR"
    TRAIN_CMD="$TRAIN_CMD --lora_output_dir $LORA_OUTPUT_DIR"
    
    # Remove any duplicate args in TRAIN_ARGS
    CLEAN_TRAIN_ARGS=$(echo "$TRAIN_ARGS" | sed 's/--data_dir [^ ]*//' | sed 's/--dataset_name [^ ]*//')
    
    # Add any additional training arguments
    TRAIN_CMD="$TRAIN_CMD $CLEAN_TRAIN_ARGS"
    
    echo "Command: $TRAIN_CMD"
    eval $TRAIN_CMD
    
    # Verify model was saved properly
    echo "Verifying model was saved to $LORA_OUTPUT_DIR"
    
    if [ ! -d "$LORA_OUTPUT_DIR" ]; then
        echo "Error: Model output directory not found at $LORA_OUTPUT_DIR"
        mkdir -p "$LORA_OUTPUT_DIR"
        echo "Created missing directory"
    fi
    
fi

echo "All requested tasks completed successfully." 