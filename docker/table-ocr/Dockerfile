FROM rust:1.81-slim-bookworm AS chef
RUN apt-get update -y && \
    apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    g++ \
    curl \
    libglib2.0-dev \
    python3 \
    python3-pip
RUN cargo install cargo-chef
WORKDIR /app

FROM chef AS planner
COPY services/paddle-proxy/ .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json
# Build application
COPY services/paddle-proxy/ .
RUN cargo build --release

FROM registry.baidubce.com/paddlex/paddlex:paddlex3.0.0b1-paddlepaddle3.0.0b1-gpu-cuda12.3-cudnn9.0-trt8.6 AS runtime
WORKDIR /app

COPY --from=builder /app/target/release/paddle-proxy /app/paddle-proxy
COPY services/paddle-proxy/config /app/config

RUN paddlex --pipeline /app/config/table_recognition.yaml || true

EXPOSE 8000

CMD ["./paddle-proxy", "--pipeline", "/app/config/table_recognition.yaml"]