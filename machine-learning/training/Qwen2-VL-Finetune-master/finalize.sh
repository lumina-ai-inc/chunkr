# Ensure necessary environment variables are set (like HF_TOKEN if needed and not logged in)
# export HF_TOKEN="YOUR_HF_TOKEN" # Uncomment and set if needed

# Define variables (optional, but makes it cleaner, mimicking run_pipeline.sh)
TRAINING_OUTPUT_DIR="output/sophris_table_vllm_fin_test"
MERGED_OUTPUT_DIR="output/sophris_table_vllm_fin_test_merged"
TARGET_HUB_REPO="ChunkrAI/chunkr-table-v1-qwen2_5VL-3B"
# Retrieve token from environment if set, otherwise script will try CLI login
# Load HF_TOKEN from .env file if it exists
if [ -f .env ]; then
    source .env
    source .env
    HF_TOKEN_VALUE=${HF_TOKEN:-""}
    echo "HF_TOKEN_VALUE: $HF_TOKEN_VALUE"
fi

# Construct the uv run command
COMMAND=(
    "uv" "run" "finalize.py"
    "--merged_model_dir" "$MERGED_OUTPUT_DIR"
    "--hub_repo_id" "$TARGET_HUB_REPO"
    "--hf_token" "$HF_TOKEN_VALUE"
)

# Conditionally add the token argument if the variable is set
if [ -n "$HF_TOKEN_VALUE" ]; then
    COMMAND+=("--hf_token" "$HF_TOKEN_VALUE")
fi

# Execute the command
"${COMMAND[@]}"
