# Chunkr Python Client

This is the Python client for the Chunkr API. It provides a simple interface to interact with Chunkr's services.

## Installation

```bash
pip install chunkr-ai
```

## Usage

We provide two clients: `Chunkr` for synchronous operations and `ChunkrAsync` for asynchronous operations.

### Synchronous Usage

```python
from chunkr_ai import Chunkr

# Initialize client
chunkr = Chunkr()

# Upload a file and wait for processing
task = chunkr.upload("document.pdf")

# Print the response
print(task)

# Get output from task
output = task.output

# If you want to upload without waiting for processing
task = chunkr.start_upload("document.pdf")
# ... do other things ...
task.poll()  # Check status when needed
```

### Asynchronous Usage

```python
from chunkr_ai import ChunkrAsync

async def process_document():
    # Initialize client
    chunkr = ChunkrAsync()

    # Upload a file and wait for processing
    task = await chunkr.upload("document.pdf")

    # Print the response
    print(task)

    # Get output from task
    output = task.output

    # If you want to upload without waiting for processing
    task = await chunkr.start_upload("document.pdf")
    # ... do other things ...
    await task.poll_async()  # Check status when needed
```

### Additional Features

Both clients support various input types:

```python
# Upload from file path
chunkr.upload("document.pdf")

# Upload from opened file
with open("document.pdf", "rb") as f:
    chunkr.upload(f)

# Upload from URL
chunkr.upload("https://example.com/document.pdf")

# Upload from base64 string
chunkr.upload("data:application/pdf;base64,JVBERi0xLjcKCjEgMCBvYmo...")

# Upload an image
from PIL import Image
img = Image.open("photo.jpg")
chunkr.upload(img)
```

### Configuration

You can provide your API key and URL in several ways:
1. Environment variables: `CHUNKR_API_KEY` and `CHUNKR_URL`
2. `.env` file
3. Direct initialization:
```python
chunkr = Chunkr(
    api_key="your-api-key",
    url="https://api.chunkr.ai"
)
```

