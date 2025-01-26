# Load Test

This is a load test for the Chunkr AI API. It uses the Chunkr Python SDK to upload files to the API and then processes them.

## Run

**Recommended approach (run in separate terminals):**

Terminal 1 - Start processors and writer:
```sh
honcho start -c processor=4
```

Terminal 2 - Run orchestrator:
```sh
uv run orchestrator.py
```

### Command Line Arguments

The orchestrator accepts the following arguments:

- `--input`: Input folder path (default: "./input")
- `--max-files`: Maximum number of files to process (default: all files)

Example:
```sh
uv run orchestrator.py --input ./my-files --max-files 100
```

## Environment Variables

```sh
CHUNKR_URL=https://api.chunkr.ai
CHUNKR_API_KEY=your_api_key
PROCESSING_QUEUE=processing_queue
WRITER_QUEUE=writer_queue
```