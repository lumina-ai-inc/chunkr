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
RUN cargo chef cook --release --recipe-path recipe.json
# Build application
COPY core/ .
RUN cargo build --release

FROM debian:bookworm-slim AS runtime
WORKDIR /app

RUN apt-get update -y; \
    apt-get install -y \
    pkg-config \
    build-essential\
    libssl-dev \
    libpq-dev \
    ca-certificates \
    libreoffice \
    file \
    imagemagick \
    ; \
    mkdir -p /app/tmp && \
    sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml

COPY --from=builder /app/target/release/core /app/core

EXPOSE 8000
ENTRYPOINT ["/app/core"]
