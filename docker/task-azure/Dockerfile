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
RUN cargo chef cook --release --recipe-path recipe.json --bin "task" --features "azure memory_profiling"
# Build application
COPY core/ .
RUN cargo build --release --bin "task" --features "azure memory_profiling"

FROM chef AS development
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --recipe-path recipe.json --bin "task" --features "azure memory_profiling"
# Copy source code
COPY core/ .

# Install all runtime dependencies
RUN apt-get update -y && \
    apt-get install -y \
    pkg-config \
    build-essential \
    libssl-dev \
    libpq-dev \
    ca-certificates \
    libreoffice \
    file \
    curl \
    imagemagick \
    pandoc \
    wget \
    gnupg && \
    rm -rf /var/lib/apt/lists/*

# Tweak ImageMagick policy
RUN sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="resource" name="width" value="32KP"\/>/<policy domain="resource" name="width" value="100KP"\/>/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="resource" name="height" value="32KP"\/>/<policy domain="resource" name="height" value="100KP"\/>/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="resource" name="area" value="256MP"\/>/<policy domain="resource" name="area" value="1GP"\/>/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="resource" name="memory" value="1024MiB"\/>/<policy domain="resource" name="memory" value="8GiB"\/>/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="resource" name="disk" value="2GiB"\/>/<policy domain="resource" name="disk" value="20GiB"\/>/' /etc/ImageMagick-6/policy.xml

# Download PDFium binary
RUN mkdir -p /app/pdfium-binaries && \
    curl -L https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-x64.tgz | \
    tar -xz -C /tmp && \
    cp /tmp/lib/libpdfium.so /app/pdfium-binaries/

# Install Chrome browser for headless operation
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update -y \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Create Chrome cache directory with proper permissions
RUN mkdir -p /app/chrome-cache && chmod 755 /app/chrome-cache

# Install Google Cloud SDK
RUN curl -sSL https://sdk.cloud.google.com | bash \
    && echo "source /root/google-cloud-sdk/path.bash.inc" >> /root/.bashrc \
    && echo "source /root/google-cloud-sdk/completion.bash.inc" >> /root/.bashrc

# Add gcloud to PATH
ENV PATH="/root/google-cloud-sdk/bin:$PATH"

# Set Chrome to run as root (required for container execution)
RUN echo 'export CHROME_DEVEL_SANDBOX=/opt/google/chrome/chrome-sandbox' >> /etc/environment

EXPOSE 8000
CMD ["cargo", "run", "--bin", "task", "--features", "azure memory_profiling"]

FROM ubuntu:24.04 AS runtime
WORKDIR /app

# Install main dependencies
RUN apt-get update -y && \
    apt-get install -y \
    pkg-config \
    build-essential \
    libssl-dev \
    libpq-dev \
    ca-certificates \
    libreoffice \
    file \
    curl \
    imagemagick \
    pandoc && \
    rm -rf /var/lib/apt/lists/*

# Check if pandoc is installed and print version, else print error and exit
RUN if pandoc --version >/dev/null 2>&1; then \
    echo "Pandoc version:"; \
    pandoc --version; \
    else \
    echo "Pandoc is not installed or not working!" >&2; \
    exit 1; \
    fi

# Tweak ImageMagick policy
RUN sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="resource" name="width" value="32KP"\/>/<policy domain="resource" name="width" value="100KP"\/>/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="resource" name="height" value="32KP"\/>/<policy domain="resource" name="height" value="100KP"\/>/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="resource" name="area" value="256MP"\/>/<policy domain="resource" name="area" value="1GP"\/>/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="resource" name="memory" value="1024MiB"\/>/<policy domain="resource" name="memory" value="8GiB"\/>/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="resource" name="disk" value="2GiB"\/>/<policy domain="resource" name="disk" value="20GiB"\/>/' /etc/ImageMagick-6/policy.xml

# Download and extract PDFium binary
RUN mkdir -p /app/pdfium-binaries && \
    curl -L https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-x64.tgz | \
    tar -xz -C /tmp && \
    cp /tmp/lib/libpdfium.so /app/pdfium-binaries/

# Copy your built Rust binary in
COPY --from=builder /app/target/release/task /app/task

# Create Chrome cache directory with proper permissions
RUN mkdir -p /app/chrome-cache && chmod 755 /app/chrome-cache

# Install Chrome browser for headless operation
RUN apt-get update -y && \
    apt-get install -y \
    wget \
    gnupg \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update -y \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN curl -sSL https://sdk.cloud.google.com | bash \
    && echo "source /root/google-cloud-sdk/path.bash.inc" >> /root/.bashrc \
    && echo "source /root/google-cloud-sdk/completion.bash.inc" >> /root/.bashrc

# Add gcloud to PATH
ENV PATH="/root/google-cloud-sdk/bin:$PATH"

# Set Chrome to run as root (required for container execution)
RUN echo 'export CHROME_DEVEL_SANDBOX=/opt/google/chrome/chrome-sandbox' >> /etc/environment

EXPOSE 8000
ENTRYPOINT ["/app/task"]