FROM rust:1.87-slim-bookworm AS chef
RUN apt-get update -y && apt-get -y install pkg-config libssl-dev libpq-dev g++ curl libglib2.0-dev
RUN cargo install cargo-chef
WORKDIR /app

FROM chef AS planner
COPY core/ .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json --bin "task"
# Build application
COPY core/ .
RUN cargo build --release --bin "task"

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
    curl \
    imagemagick \
    pandoc \
    ; \
    mkdir -p /app/tmp && \
    sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml

# Download and extract PDFium binary
RUN mkdir -p /app/pdfium-binaries && \
    curl -L https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-x64.tgz | \
    tar -xz -C /tmp && \
    cp /tmp/lib/libpdfium.so /app/pdfium-binaries/

COPY --from=builder /app/target/release/task /app/task

EXPOSE 8000
ENTRYPOINT ["/app/task"]