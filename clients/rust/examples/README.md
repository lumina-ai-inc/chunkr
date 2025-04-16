# Chunkr API Rust Client Examples

This directory contains examples showing how to use the Chunkr API with the Rust client.

## Setup

1. First, make sure you have the required dependencies in your `Cargo.toml`:

```bash
cargo add chunkr-ai dotenvy config
```

2. Copy `.env.example` to `.env` and fill in your Chunkr API key:

```bash
cp .env.example .env
# Edit .env with your API key
```

## Running the Examples

### Basic Client Usage

Shows how to initialize a client and make a simple health check request:

```bash
cargo run --example basic_usage
```

## Notes

- These examples use the `dotenvy` crate to load environment variables from a `.env` file
- The `config` crate is used to handle configuration from multiple sources
- The client is configured with default headers and timeouts

## Custom Client Configuration

You can customize the client by modifying the following:

- Connect timeout: Change the duration in `connect_timeout()`
- Request timeout: Change the duration in `timeout()`
- Default headers: Add more headers to the `HeaderMap`

For more details, refer to the main library documentation. 