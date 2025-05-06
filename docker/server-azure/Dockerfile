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
RUN cargo chef cook --release --recipe-path recipe.json --features "azure memory_profiling"
# Build application
COPY core/ .
RUN cargo build --release --features "azure memory_profiling"

FROM ubuntu:24.04 AS runtime
WORKDIR /app

# Install all deps except LibreOffice
RUN apt-get update -y && \
    apt-get install -y \
    pkg-config \
    build-essential \
    libssl-dev \
    libpq-dev \
    ca-certificates \
    file \
    curl \
    imagemagick && \
    rm -rf /var/lib/apt/lists/*

# Download & install LibreOffice 25.2.2 directly from TDF
RUN curl -fsSL -o /tmp/libreoffice.tar.gz \
    https://download.documentfoundation.org/libreoffice/stable/25.2.2/deb/x86_64/LibreOffice_25.2.2_Linux_x86-64_deb.tar.gz && \
    tar -xzf /tmp/libreoffice.tar.gz -C /tmp && \
    dpkg -i /tmp/LibreOffice_*/DEBS/*.deb && \
    rm -rf /tmp/LibreOffice_* /tmp/libreoffice.tar.gz

# Tweak ImageMagick policy
RUN sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml

# Download and extract PDFium binary
RUN mkdir -p /app/pdfium-binaries && \
    curl -L https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-x64.tgz | \
    tar -xz -C /tmp && \
    cp /tmp/lib/libpdfium.so /app/pdfium-binaries/

# Copy your built Rust binary in
COPY --from=builder /app/target/release/core /app/core

EXPOSE 8000
ENTRYPOINT ["/app/core"]