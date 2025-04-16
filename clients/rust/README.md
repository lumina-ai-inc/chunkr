# Chunkr API Rust Client

This is a Rust client for the Chunkr API, generated using [Progenitor](https://github.com/oxidecomputer/progenitor).

## Prerequisites

Before using this client, you need to generate the OpenAPI specification for the Chunkr API:

```bash
# Generate the OpenAPI spec (save to ~/.chunkr/openapi.json)
cargo run --bin generate_openapi

# Or specify a custom output path
cargo run --bin generate_openapi -- --output /path/to/openapi.json
```

## Usage

Add the client to your `Cargo.toml`:

```toml
[dependencies]
chunkr-client = { git = "https://github.com/yourusername/chunkr", path = "clients/rust" }
```

Basic example:

```rust
use chunkr_client;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment variable
    let api_key = env::var("CHUNKR_API_KEY").expect("CHUNKR_API_KEY not set");
    
    // Create client
    let client = chunkr_client::create_client("https://api.chunkr.ai", api_key);
    
    // Make API calls
    let health = client.health_check().await?;
    println!("API health: {}", health.status());
    
    Ok(())
}
```

## Examples

Run the basic usage example:

```bash
CHUNKR_API_KEY=your_api_key cargo run --example basic_usage
```

## Features

This client supports all endpoints defined in the Chunkr API:

- Health check
- Task management (create, read, update, delete)
- Tasks listing
- And more based on the OpenAPI specification 