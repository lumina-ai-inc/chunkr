FROM rust:1.76-slim-bookworm AS planner
RUN apt-get update -y && apt-get -y install pkg-config libssl-dev libpq-dev g++ curl
RUN cargo install cargo-chef
WORKDIR /app
COPY chunkmydocs/ .
RUN cargo chef prepare --recipe-path recipe.json

FROM rust:1.76-slim-bookworm AS builder
RUN apt-get update -y && apt-get -y install pkg-config libssl-dev libpq-dev g++ curl
RUN cargo install cargo-chef
WORKDIR /app
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json
# Build application
COPY chunkmydocs/ .
RUN cargo build --release --bin task_processor

FROM debian:bookworm-slim as runtime
WORKDIR /app
COPY --from=builder /app/target/release/task_processor /app/task_processor

EXPOSE 8000
ENTRYPOINT ["/app/task_processor"]