#!/bin/bash

# Default values
NUM_PDFS=200
NUM_WORKERS=7
DATASET_NAME="tables-vlm-azure-distill-v1"
FLUSH_REDIS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pdfs=*)
            NUM_PDFS="${1#*=}"
            ;;
        --workers=*)
            NUM_WORKERS="${1#*=}"
            ;;
        --dataset=*)
            DATASET_NAME="${1#*=}"
            ;;
        --flush)
            FLUSH_REDIS=true
            ;;
        --help)
            echo "Usage: $0 [--pdfs=N] [--workers=N] [--dataset=NAME] [--flush]"
            echo ""
            echo "Options:"
            echo "  --pdfs=N       Number of PDFs to queue (default: 5)"
            echo "  --workers=N    Number of Chunkr worker processes (default: 2)"
            echo "  --dataset=NAME Dataset name (default: tables-vlm-azure-distill-v1)"
            echo "  --flush        Flush all Redis data before processing"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
    shift
done

# Make sure the directory structure exists
mkdir -p redis-data

# Display configuration
echo "Starting processing with:"
echo "  Dataset: $DATASET_NAME"
echo "  PDFs to queue: $NUM_PDFS"
echo "  Worker processes: $NUM_WORKERS"
echo ""

# Start Redis in its own window/tab
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    osascript -e 'tell app "Terminal" to do script "cd '$PWD' && uv run processors/start_redis.py"'
elif command -v gnome-terminal &> /dev/null; then
    # Linux with GNOME
    gnome-terminal -- bash -c "cd '$PWD' && uv run processors/start_redis.py; exec bash"
elif command -v xterm &> /dev/null; then
    # Linux with X11
    xterm -e "cd '$PWD' && uv run processors/start_redis.py" &
else
    # Fallback - run in background
    uv run processors/start_redis.py &
    REDIS_PID=$!
    echo "Started Redis with PID $REDIS_PID"
fi

# Wait for Redis to start
echo "Waiting for Redis to start..."
sleep 3

# Check if Redis is running
if ! uv run processors/pdf_queue.py redis status; then
    echo "Redis failed to start. Exiting."
    exit 1
fi

# Show stats in another window/tab
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    osascript -e 'tell app "Terminal" to do script "cd '$PWD' && uv run processors/pdf_queue.py stats --refresh 2"'
elif command -v gnome-terminal &> /dev/null; then
    # Linux with GNOME
    gnome-terminal --title="Queue Stats" -- bash -c "cd '$PWD' && uv run processors/pdf_queue.py stats --refresh 2; exec bash"
elif command -v xterm &> /dev/null; then
    # Linux with X11
    xterm -title "Queue Stats" -e "cd '$PWD' && uv run processors/pdf_queue.py stats --refresh 2" &
else
    # Fallback - no stats window
    echo "Starting stats monitor in background..."
    uv run processors/pdf_queue.py stats --once
fi

# Create a temporary script file for the watch command
WATCH_SCRIPT="/tmp/s3_watch_script_$$.sh"
cat > "$WATCH_SCRIPT" << 'EOL'
#!/bin/bash
folders=(
  "chunkr_outputs"
  "pdfs"
  "raw-pdfs"
  "table_html"
  "table_mkd"
  "tables"
  "table_images"
)
bucket="s3://chunkr-datasets/DATASET_NAME_PLACEHOLDER"

printf "\033c"  # clear screen
printf "%-20s %6s\n" "S3 Folder" "Count"
printf "%-20s %6s\n" "--------------------" "------"

for folder in "${folders[@]}"; do
  count=$(aws s3 ls "$bucket/$folder/" --recursive --summarize 2>/dev/null | grep "Total Objects:" | awk "{print \$3}")
  printf "%-20s %6s\n" "$folder" "${count:-0}"
done
EOL

# Replace dataset name placeholder
sed -i "s|DATASET_NAME_PLACEHOLDER|$DATASET_NAME|g" "$WATCH_SCRIPT"
chmod +x "$WATCH_SCRIPT"

# Start the S3 watch in another window/tab
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    osascript -e 'tell app "Terminal" to do script "cd '$PWD' && watch -n 5 '"$WATCH_SCRIPT"'"'
elif command -v gnome-terminal &> /dev/null; then
    # Linux with GNOME
    gnome-terminal --title="S3 Counts" -- bash -c "cd '$PWD' && watch -n 5 $WATCH_SCRIPT; exec bash"
elif command -v xterm &> /dev/null; then
    # Linux with X11
    xterm -title "S3 Counts" -e "cd '$PWD' && watch -n 5 $WATCH_SCRIPT" &
else
    # Fallback - run once
    echo "S3 folder counts (one-time):"
    bash "$WATCH_SCRIPT"
fi

# Queue PDFs
if [ "$FLUSH_REDIS" = true ]; then
    echo "Flushing all Redis data..."
    redis-cli FLUSHALL
    echo "Redis data cleared."
fi

echo "Queueing $NUM_PDFS PDFs..."
uv run processors/pdf_queue.py queue --count $NUM_PDFS --dataset "$DATASET_NAME"

# Start workers
echo "Starting workers..."
uv run processors/chunkr.py worker --workers=$NUM_WORKERS

echo "Processing complete."

# Clean up the temporary script
rm -f "$WATCH_SCRIPT" 