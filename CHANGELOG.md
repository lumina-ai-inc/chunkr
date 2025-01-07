# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `chunk_processing` config to control chunking
- Added `high_resolution` config to control image density
- Added `segmentation_processing` config to control LLM processing on the segments
- Added `segmentation_strategy` to control segmentation
- Concurrent OCR and segmentation
- Concurrent page processing

### Changed
- Combined all workers into a `task` worker. See [279](https://github.com/lumina-ai-inc/chunkr/issues/279)

### Fixed
- List items incorrect heuristics. See [276](https://github.com/lumina-ai-inc/chunkr/issues/276)
- Reading order

### Removed
(All changes maintain compatibility with old configs)
- Deprecated `model` config
- Deprecated `target_chunk_length`, you can now use `chunk_processing.target_length` instead 
- Deprecated `structured_extraction.json_schema.type`
- Deprecated `ocr_strategy.Off`
