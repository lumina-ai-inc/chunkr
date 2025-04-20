# Ensure necessary environment variables are set (like HF_TOKEN if needed and not logged in)
# export HF_TOKEN="YOUR_HF_TOKEN" # Uncomment and set if needed

# Define variables (optional, but makes it cleaner, mimicking run_pipeline.sh)
TRAINING_OUTPUT_DIR="output/sophris_table_prod1"
MERGED_OUTPUT_DIR="output/sophris_table_prod1_final"
TARGET_HUB_REPO="ChunkrAI/sophris-table-qwen2_5VL-3B"
# Retrieve token from environment if set, otherwise script will try CLI login
HF_TOKEN_VALUE=${HF_TOKEN:-""}

# Construct the uv run command
COMMAND=(
    "uv" "run" "finalize.py"
    "--lora_adapter_path" "$TRAINING_OUTPUT_DIR"
    "--output_dir" "$MERGED_OUTPUT_DIR"
    "--hub_repo_id" "$TARGET_HUB_REPO"
    "--safe_serialization"
)

# Conditionally add the token argument if the variable is set
if [ -n "$HF_TOKEN_VALUE" ]; then
    COMMAND+=("--hf_token" "$HF_TOKEN_VALUE")
fi

# Execute the command
"${COMMAND[@]}"
