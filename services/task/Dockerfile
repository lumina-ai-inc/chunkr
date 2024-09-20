# Start with a base image, for example, Ubuntu 20.04
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    autoconf \
    pkg-config \
    build-essential \
    curl \
    libpng-dev \
    ghostscript \
    libgs-dev \
    libpdf-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install ImageMagick
RUN wget https://github.com/ImageMagick/ImageMagick/archive/refs/tags/7.1.0-38.tar.gz \
    && tar xzf 7.1.0-38.tar.gz \
    && rm 7.1.0-38.tar.gz \
    && cd ImageMagick-7.1.0-38 \
    && ./configure --prefix=/usr/local \
        --with-bzlib=yes \
        --with-fontconfig=yes \
        --with-freetype=yes \
        --with-gslib=yes \
        --with-gvc=yes \
        --with-jpeg=yes \
        --with-jp2=yes \
        --with-png=yes \
        --with-tiff=yes \
        --with-xml=yes \
        --with-gs-font-dir=yes \
        --with-gslib=yes \
        --with-pdf=yes \
    && make -j \
    && make install \
    && ldconfig /usr/local/lib \
    && cd .. \
    && rm -rf ImageMagick-7.1.0-38

# Set the working directory
WORKDIR /app

