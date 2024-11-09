This is holds the core of the chunkr api. The API consists of multiple services that work together to extract structured data from documents.

# Development

## Setup

```bash
# Install Rust and Cargo
curl https://sh.rustup.rs -sSf | sh

# Source the cargo environment
source $HOME/.cargo/env

# Copy the example env file
cp .env.example .env
```

## Running the Services

### Rust services

```bash
# Start the server
cargo run

# Start the workers
## Preprocess: Converts documents to PDF if needed, counts pages, and convert the pdf pages to images
cargo run --bin preprocess

## Segmentation: Uses a layout model to segment the pages into chunks
cargo run --bin high-quality
cargo run --bin fast

## Postprocess: Crops the segments using the bounding boxes from the layout model and creates chunks from the segments
cargo run --bin postprocess

## OCR: Uses a OCR model to extract text from the images, and convert tables in to HTML
cargo run --bin ocr

## Structured Extraction: Uses a LLM to extract structured json from the chunks
cargo run --bin structured-extraction
```

### Other Services

To run the other services it is recommended to use the docker compose file in the root of the repo.
For Docker Compose setup and usage instructions, please refer to [Quick Start with Docker Compose](../README.md#quick-start-with-docker-compose).

Set replicas to 0 for services you don't want to run/aren't actively being worked on.
