FROM rust:1.82-slim-bookworm AS chef
RUN apt-get update -y && apt-get -y install pkg-config libssl-dev libpq-dev g++ curl libglib2.0-dev
RUN cargo install cargo-chef
WORKDIR /app

FROM chef AS planner
COPY core/ .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json --bin "task" --features azure
# Build application
COPY core/ .
RUN cargo build --release --bin "task" --features azure

FROM debian:bookworm-slim AS runtime
WORKDIR /app

COPY subtrace.yaml .

RUN apt-get update -y; \
    apt-get install -y \
    pkg-config \
    build-essential\
    libssl-dev \
    libpq-dev \
    ca-certificates \
    libreoffice \
    file \
    curl \
    imagemagick \
    ; \
    mkdir -p /app/tmp && \
    curl -fsSLO https://subtrace.dev/download/latest/$(uname -s)/$(uname -m)/subtrace && \
    chmod +x ./subtrace && \
    sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml

COPY --from=builder /app/target/release/task /app/task
COPY subtrace.yaml .

EXPOSE 8000
ENTRYPOINT ["/app/task"]
