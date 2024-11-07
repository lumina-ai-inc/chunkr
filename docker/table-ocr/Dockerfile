FROM registry.baidubce.com/paddlex/paddlex:paddlex3.0.0b1-paddlepaddle3.0.0b1-gpu-cuda12.3-cudnn9.0-trt8.6 AS chef
# Install Rust build dependencies
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

# Install Rust and cargo-chef
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
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

FROM chef AS runtime
WORKDIR /app

COPY --from=builder /app/target/release/paddle-proxy /app/paddle-proxy
COPY services/paddle-proxy/config /app/config

EXPOSE 8000

CMD ["/app/paddle-proxy", "--pipeline", "/app/config/table_recognition.yaml"]