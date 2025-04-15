# Chunkr Python Client

This provides a simple interface to interact with the Chunkr API.

## Getting Started

You can get an API key from [Chunkr](https://chunkr.ai) or deploy your own Chunkr instance. For self-hosted deployment options, check out our [deployment guide](https://github.com/lumina-ai-inc/chunkr/tree/main?tab=readme-ov-file#self-hosted-deployment-options).

For more information about the API and its capabilities, visit the [Chunkr API docs](https://docs.chunkr.ai).

## Installation

```bash
pip install chunkr-ai
```

## Usage

The `Chunkr` client works seamlessly in both synchronous and asynchronous contexts.

### Synchronous Usage

```python
from chunkr_ai import Chunkr

# Initialize client
chunkr = Chunkr()

# Upload a file and wait for processing
task = chunkr.upload("document.pdf")
print(task.task_id)

# Create task without waiting
task = chunkr.create_task("document.pdf")
result = task.poll()  # Check status when needed

# Clean up when done
chunkr.close()
```

### Asynchronous Usage

```python
from chunkr_ai import Chunkr
import asyncio

async def process_document():
    # Initialize client
    chunkr = Chunkr()

    try:
        # Upload a file and wait for processing
        task = await chunkr.upload("document.pdf")
        print(task.task_id)

        # Create task without waiting
        task = await chunkr.create_task("document.pdf")
        result = await task.poll()  # Check status when needed
    finally:
        await chunkr.close()

# Run the async function
asyncio.run(process_document())
```

### Concurrent Processing

The client supports both async concurrency and multiprocessing:

```python
# Async concurrency
async def process_multiple():
    chunkr = Chunkr()
    try:
        tasks = [
            chunkr.upload("doc1.pdf"),
            chunkr.upload("doc2.pdf"),
            chunkr.upload("doc3.pdf")
        ]
        results = await asyncio.gather(*tasks)
    finally:
        await chunkr.close()

# Multiprocessing
from multiprocessing import Pool

def process_file(path):
    chunkr = Chunkr()
    try:
        return chunkr.upload(path)
    finally:
        chunkr.close()

with Pool(processes=3) as pool:
    results = pool.map(process_file, ["doc1.pdf", "doc2.pdf", "doc3.pdf"])
```

### Input Types

The client supports various input types:

```python
# File path
chunkr.upload("document.pdf")

# Opened file
with open("document.pdf", "rb") as f:
    chunkr.upload(f)

# PIL Image
from PIL import Image
img = Image.open("photo.jpg")
chunkr.upload(img)
```

### Configuration

You can customize the processing behavior by passing a `Configuration` object:

```python
from chunkr_ai.models import (
    Configuration, 
    OcrStrategy, 
    SegmentationStrategy, 
    GenerationStrategy
)

config = Configuration(
    ocr_strategy=OcrStrategy.AUTO,
    segmentation_strategy=SegmentationStrategy.LAYOUT_ANALYSIS,
    high_resolution=True,
    expires_in=3600,  # seconds
)

# Works in both sync and async contexts
task = chunkr.upload("document.pdf", config)  # sync
task = await chunkr.upload("document.pdf", config)  # async
```

#### Available Configuration Examples

- **Chunk Processing**
  ```python
  from chunkr_ai.models import ChunkProcessing
  config = Configuration(
      chunk_processing=ChunkProcessing(target_length=1024)
  )
  ```
- **Expires In**
  ```python
  config = Configuration(expires_in=3600)
  ```

- **High Resolution**
  ```python
  config = Configuration(high_resolution=True)
  ```

- **JSON Schema**
  ```python
  config = Configuration(json_schema=JsonSchema(
      title="Sales Data",
      properties=[
          Property(name="Person with highest sales", prop_type="string", description="The person with the highest sales"),
          Property(name="Person with lowest sales", prop_type="string", description="The person with the lowest sales"),
      ]
  ))
  ```

- **OCR Strategy**
  ```python
  config = Configuration(ocr_strategy=OcrStrategy.AUTO)
  ```

- **Segment Processing**
  ```python
  from chunkr_ai.models import SegmentProcessing, GenerationConfig, GenerationStrategy
  config = Configuration(
      segment_processing=SegmentProcessing(
          page=GenerationConfig(
              html=GenerationStrategy.LLM,
              markdown=GenerationStrategy.LLM
          )
      )
  )
  ```

- **Segmentation Strategy**
  ```python
  config = Configuration(
      segmentation_strategy=SegmentationStrategy.LAYOUT_ANALYSIS  # or SegmentationStrategy.PAGE
  )
  ```

## Environment Setup

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

## Resource Management

It's recommended to properly close the client when you're done:

```python
# Sync context
chunkr = Chunkr()
try:
    result = chunkr.upload("document.pdf")
finally:
    chunkr.close()

# Async context
async def process():
    chunkr = Chunkr()
    try:
        result = await chunkr.upload("document.pdf")
    finally:
        await chunkr.close()
```