# Changelog

All notable changes to the Chunkr Java Client will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-30

### Added
- Initial release of the Chunkr Java Client
- Core `ChunkrClient` class with full API support
- Complete model classes for all API responses
- Support for multiple input types:
  - File paths
  - URLs (HTTP/HTTPS)
  - Byte arrays
  - InputStreams
  - File objects
  - Base64 encoded strings
- Configuration system with builder pattern
- Comprehensive error handling
- Async task processing with polling
- Task management operations (create, get, update, delete, cancel)
- Content extraction in multiple formats (HTML, Markdown, raw text)
- Environment variable support for API key and base URL
- Complete documentation and examples
- Unit tests for core functionality
- Maven build configuration with proper dependencies

### Features
- **Document Processing**: Upload and process PDFs, Word docs, PowerPoint, images
- **Layout Analysis**: Intelligent document structure detection
- **OCR Support**: Extract text with bounding boxes and confidence scores
- **Multiple Output Formats**: Get results as HTML, Markdown, or raw text
- **Configurable Processing**: Fine-tune OCR, segmentation, and generation strategies
- **Async Operations**: Non-blocking task creation and polling
- **Resource Management**: Proper HTTP client lifecycle management
- **Self-hosted Support**: Compatible with self-hosted Chunkr instances

### Dependencies
- OkHttp 4.11.0 for HTTP operations
- Jackson 2.15.2 for JSON processing
- SLF4J 2.0.7 for logging
- JUnit 5.10.0 for testing

### Requirements
- Java 11 or higher
- Maven 3.6+ or Gradle 6.0+