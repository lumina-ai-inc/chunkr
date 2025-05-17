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
## Each worker is 1 task
cargo run --bin task
```

### Other Services

To run the other services it is recommended to use the docker compose file in the root of the repo.
For Docker Compose setup and usage instructions, please refer to [Quick Start with Docker Compose](../README.md#quick-start-with-docker-compose).

Set replicas to 0 for services you don't want to run/aren't actively being worked on.

