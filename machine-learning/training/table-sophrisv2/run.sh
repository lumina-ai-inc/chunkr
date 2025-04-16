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
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
BATCH_SIZE=2
GRAD_ACCUM=4
MAX_STEPS=100
LR=1e-5
NUM_GPUS=1
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
SEED=42
EVAL_SPLIT="test"
NUM_EVAL_SAMPLES=10
DATA_LIMIT=""
OUTPUT_DIR="outputs/qwen2_5_Vl_3B-table-finetune"
CACHE_DIR="model_cache"  # Cache directory for downloaded models
ZERO_STAGE=3
PORT=29919
DYNAMO_BACKEND="no"

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
    echo "  --output-dir DIR        Output directory (default: $OUTPUT_DIR)"
    echo "  --batch-size NUM        Batch size per device (default: $BATCH_SIZE)"
    echo "  --grad-accum NUM        Gradient accumulation steps (default: $GRAD_ACCUM)"
    echo "  --max-steps NUM         Maximum training steps (default: $MAX_STEPS)"
    echo "  --lr NUM                Learning rate (default: $LR)"
    echo "  --num-gpus NUM          Number of GPUs to use (default: $NUM_GPUS)"
    echo "  --zero-stage NUM        DeepSpeed ZeRO optimization level (default: $ZERO_STAGE)"
    echo "  --port NUM              Main process port for distributed training (default: $PORT)"
    echo ""
    echo "Evaluation options:"
    echo "  --eval-split TYPE       Split to use for evaluation (val or test, default: $EVAL_SPLIT)"
    echo "  --num-eval-samples NUM  Number of samples to evaluate (default: $NUM_EVAL_SAMPLES)"
    echo "  --skip-baseline         Skip baseline model evaluation (default: false)"
    echo ""
    echo "Common options:"
    echo "  --seed NUM              Random seed (default: $SEED)"
    echo "  --cuda-devices LIST     Comma-separated list of CUDA devices to use (default: all available)"
    echo "  -h, --help              Show this help message"
    exit 1
}

if [ "$#" -eq 0 ]; then
    show_help
fi

# Process command line arguments
CUDA_DEVICES=""
TRAIN_ARGS=""
PREPARE_ARGS=""
EVAL_ARGS=""
SKIP_BASELINE=false
BASELINE_MODELS=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --prepare) PREPARE=true; shift ;;
        --train) TRAIN=true; shift ;;
        --eval) EVAL=true; shift ;;
        --all) PREPARE=true; TRAIN=true; EVAL=true; shift ;;
        
        # Data preparation options
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --data-limit) DATA_LIMIT="$2"; PREPARE_ARGS="$PREPARE_ARGS --data_limit $2"; shift 2 ;;
        --train-ratio) TRAIN_RATIO="$2"; PREPARE_ARGS="$PREPARE_ARGS --train_ratio $2"; shift 2 ;;
        --val-ratio) VAL_RATIO="$2"; PREPARE_ARGS="$PREPARE_ARGS --val_ratio $2"; shift 2 ;;
        --test-ratio) TEST_RATIO="$2"; PREPARE_ARGS="$PREPARE_ARGS --test_ratio $2"; shift 2 ;;
        --s3-bucket) S3_BUCKET="$2"; PREPARE_ARGS="$PREPARE_ARGS --s3_bucket $2"; shift 2 ;;
        --dataset-name) DATASET_NAME="$2"; shift 2 ;;
        
        # Training options
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --zero-stage) ZERO_STAGE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        
        # Evaluation options
        --eval-split) EVAL_SPLIT="$2"; EVAL_ARGS="$EVAL_ARGS --eval_split $2"; shift 2 ;;
        --num-eval-samples) NUM_EVAL_SAMPLES="$2"; EVAL_ARGS="$EVAL_ARGS --num_samples $2"; shift 2 ;;
        --skip-baseline) SKIP_BASELINE=true; EVAL_ARGS="$EVAL_ARGS --skip_baseline"; shift ;;
        
        # Common options
        --seed) SEED="$2"; PREPARE_ARGS="$PREPARE_ARGS --seed $2"; TRAIN_ARGS="$TRAIN_ARGS --seed $2"; EVAL_ARGS="$EVAL_ARGS --seed $2"; shift 2 ;;
        --cuda-devices) CUDA_DEVICES="$2"; shift 2 ;;
        --help|-h) show_help ;;
        
        *) echo "Unknown option: $1"; show_help ;;
    esac
done

# Check if at least one action was specified
if [ "$PREPARE" = false ] && [ "$TRAIN" = false ] && [ "$EVAL" = false ]; then
    echo "Error: At least one action (--prepare, --train, --eval, or --all) must be specified."
    show_help
fi

# Auto-detect CUDA devices if not specified
if [ -z "$CUDA_DEVICES" ]; then
    # Count available GPUs and generate comma-separated list
    if command -v nvidia-smi &> /dev/null; then
        NUM_AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        if [ "$NUM_AVAILABLE_GPUS" -gt 0 ]; then
            # Generate comma-separated GPU indices
            CUDA_DEVICES=$(seq -s, 0 $((NUM_AVAILABLE_GPUS - 1)))
            echo "Auto-detected $NUM_AVAILABLE_GPUS GPUs: $CUDA_DEVICES"
            
            # If num_gpus wasn't explicitly set, use auto-detected value
            if [ "$NUM_GPUS" -eq 1 ] && [ "$NUM_AVAILABLE_GPUS" -gt 1 ]; then
                NUM_GPUS=$NUM_AVAILABLE_GPUS
                echo "Setting NUM_GPUS to auto-detected value: $NUM_GPUS"
            fi
        else
            echo "No GPUs detected by nvidia-smi"
            CUDA_DEVICES="0"  # Default to single GPU
        fi
    else
        echo "nvidia-smi not found, defaulting to single GPU mode"
        CUDA_DEVICES="0"
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Make sure cache directory exists
CACHE_DIR=$(realpath "$CACHE_DIR")
mkdir -p "$CACHE_DIR"

# Determine model directory for downloading
MODEL_DIR="$CACHE_DIR/models/$MODEL_NAME"
mkdir -p "$MODEL_DIR"

# Step 0: Download Model if needed
echo "===== STEP 0: CHECKING FOR MODEL ====="
if [ -f "$MODEL_DIR/config.json" ]; then
    echo "Model already exists at $MODEL_DIR"
else
    echo "Downloading model $MODEL_NAME to $MODEL_DIR"
    
    # Use Python to download the model with HuggingFace
    python -c "
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_name = '$MODEL_NAME'
cache_dir = '$MODEL_DIR'

# Ensure the target directory exists
os.makedirs(cache_dir, exist_ok=True)

try:
    logger.info(f'Downloading processor for {model_name} to {cache_dir}...')
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, resume_download=True, trust_remote_code=True)
    processor.save_pretrained(cache_dir)
    logger.info(f'Processor downloaded and saved to {cache_dir}')

    logger.info(f'Downloading model {model_name} to {cache_dir}...')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        resume_download=True,
        trust_remote_code=True
    )
    logger.info(f'Model files downloaded and configured in {cache_dir}')

except Exception as e:
    logger.error(f'Error during model download: {e}')
    # Exit with a non-zero status code to signal failure to the shell script
    import sys
    sys.exit(1)
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

# Local Model Path determination
LOCAL_MODEL_PATH=$(realpath "$MODEL_DIR")
if [ ! -d "$LOCAL_MODEL_PATH" ] || [ ! -f "$LOCAL_MODEL_PATH/config.json" ]; then
    echo "Warning: Pre-downloaded model not found or incomplete at $LOCAL_MODEL_PATH."
    LOCAL_MODEL_PATH=$MODEL_NAME
else
    echo "Found pre-downloaded base model at: $LOCAL_MODEL_PATH"
fi

# Step 1: Prepare Dataset if requested
if [ "$PREPARE" = true ]; then
    echo "===== STEP 1: PREPARING DATASET ====="
    PREPARE_CMD="python prepare_dataset.py"
    PREPARE_CMD="$PREPARE_CMD --output_dir $DATA_DIR"
    PREPARE_CMD="$PREPARE_CMD --dataset_name $DATASET_NAME"
    PREPARE_CMD="$PREPARE_CMD $PREPARE_ARGS"
    
    echo "Command: $PREPARE_CMD"
    $PREPARE_CMD
    
    # Extract dataset path from output for use in training
    PREPARED_DATASET_PATH=$(grep -oP "Output directory: \K.*" "$DATA_DIR/$DATASET_NAME/prepare.log" | tail -1)
    if [ -n "$PREPARED_DATASET_PATH" ]; then
        echo "Dataset prepared at: $PREPARED_DATASET_PATH"
        DATA_DIR="$PREPARED_DATASET_PATH"
    else
        echo "Could not determine prepared dataset path, using default: $DATA_DIR"
    fi
fi

# Step 2: Train Model if requested
if [ "$TRAIN" = true ]; then
    echo "===== STEP 2: TRAINING MODEL ====="
    
    # Set DeepSpeed and Accelerate environment
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    export ACCELERATE_DEBUG_VERBOSITY="debug"
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    
    # Base command for DeepSpeed training
    TRAIN_CMD="accelerate launch"
    TRAIN_CMD="$TRAIN_CMD --main_process_port=$PORT"
    TRAIN_CMD="$TRAIN_CMD --mixed_precision=bf16"
    TRAIN_CMD="$TRAIN_CMD --dynamo_backend=$DYNAMO_BACKEND"
    TRAIN_CMD="$TRAIN_CMD --num_machines=1"
    TRAIN_CMD="$TRAIN_CMD --num_processes=$NUM_GPUS"
    TRAIN_CMD="$TRAIN_CMD --use_deepspeed"
    TRAIN_CMD="$TRAIN_CMD trainer.py"
    
    # Add trainer parameters
    TRAIN_CMD="$TRAIN_CMD --data_dir $DATA_DIR"
    TRAIN_CMD="$TRAIN_CMD --dataset_name $DATASET_NAME"
    TRAIN_CMD="$TRAIN_CMD --model_name $MODEL_NAME"
    TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
    TRAIN_CMD="$TRAIN_CMD --gradient_accumulation_steps $GRAD_ACCUM"
    TRAIN_CMD="$TRAIN_CMD --learning_rate $LR"
    TRAIN_CMD="$TRAIN_CMD --max_steps $MAX_STEPS"
    TRAIN_CMD="$TRAIN_CMD --output_dir $OUTPUT_DIR"
    TRAIN_CMD="$TRAIN_CMD --local_model_path $LOCAL_MODEL_PATH"
    TRAIN_CMD="$TRAIN_CMD --zero_stage $ZERO_STAGE"
    TRAIN_CMD="$TRAIN_CMD $TRAIN_ARGS"
    
    echo "Command: $TRAIN_CMD"
    $TRAIN_CMD
    
    # Check if training completed successfully
    if [ $? -eq 0 ]; then
        TRAINING_SUCCESS=true
        echo "Training completed successfully."
    else
        echo "Error: Training script failed with exit code $?."
        exit 1
    fi
fi

# Step 3: Evaluate Model if requested
if [ "$EVAL" = true ]; then
    echo "===== STEP 3: EVALUATING MODEL ====="
    
    # Find the latest model directory if it exists
    LATEST_MODEL=""
    if [ -d "$OUTPUT_DIR" ]; then
        LATEST_MODEL=$(find "$OUTPUT_DIR" -maxdepth 1 -name "model_*" -type d | sort -r | head -n 1)
    fi
    
    if [ -z "$LATEST_MODEL" ]; then
        echo "Warning: No model directory found in $OUTPUT_DIR. Will use base model for evaluation."
        FINETUNED_PATH=""
    else
        echo "Found latest model directory: $LATEST_MODEL"
        FINETUNED_PATH="$LATEST_MODEL"
    fi
    
    EVAL_CMD="python eval.py"
    EVAL_CMD="$EVAL_CMD --data_dir $DATA_DIR"
    EVAL_CMD="$EVAL_CMD --dataset_name $DATASET_NAME"
    EVAL_CMD="$EVAL_CMD --base_model $MODEL_NAME"
    
    if [ -n "$FINETUNED_PATH" ]; then
        EVAL_CMD="$EVAL_CMD --finetuned_path $FINETUNED_PATH"
    fi
    
    if [ "$SKIP_BASELINE" = true ]; then
        EVAL_CMD="$EVAL_CMD --skip_baseline"
    fi
    
    EVAL_CMD="$EVAL_CMD $EVAL_ARGS"
    
    echo "Command: $EVAL_CMD"
    $EVAL_CMD
    
    if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully."
    else
        echo "Error: Evaluation script failed with exit code $?."
        exit 1
    fi
fi

echo "All requested tasks completed successfully." 