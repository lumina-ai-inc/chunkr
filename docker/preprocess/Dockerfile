FROM rust:1.81-slim-bookworm AS chef
RUN apt-get update -y && apt-get -y install pkg-config libssl-dev libpq-dev g++ curl libglib2.0-dev
RUN cargo install cargo-chef
WORKDIR /app

FROM chef AS planner
COPY chunkmydocs/ .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json --bin "preprocess"
# Build application
COPY chunkmydocs/ .
RUN cargo build --release --bin "preprocess"

FROM debian:bookworm-slim AS runtime
WORKDIR /app

RUN apt-get update -y; \
    apt-get install -y \
    pkg-config \
    build-essential\
    libssl-dev \
    libpq-dev \
    ca-certificates \
    wget && \
    mkdir -p /app/tmp

RUN wget https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-x86.tgz && \
    mkdir -p /usr/local/pdfium && \
    tar -xzvf pdfium-linux-x86.tgz -C /usr/local/pdfium && \
    rm pdfium-linux-x86.tgz

COPY --from=builder /app/target/release/preprocess /app/preprocess

EXPOSE 8000
ENTRYPOINT ["/app/preprocess"]
