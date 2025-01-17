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
    await task.poll()  # Check status when needed
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

You can customize the processing behavior by passing a `Configuration` object:

```python
from chunkr_ai.models import Configuration, OcrStrategy, SegmentationStrategy, GenerationStrategy

# Basic configuration
config = Configuration(
    ocr_strategy=OcrStrategy.AUTO,
    segmentation_strategy=SegmentationStrategy.LAYOUT_ANALYSIS,
    high_resolution=True,
    expires_in=3600,  # seconds
)

# Upload with configuration
task = chunkr.upload("document.pdf", config)
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

## Environment setup

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