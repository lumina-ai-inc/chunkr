# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added route `POST /task/parse` and `PATCH /task/{task_id}/parse` to parse a task. These routes are exactly the same as the `POST /task` and `PATCH /task/{task_id}` routes, but don't use a multipart request.
>The old routes are deprecated but will continue to work for the foreseeable future.
- Batch parallization, so individual tasks can take full advatage of unused GPU resources.

### Changed
- OCR `All` is now the default strategy
- Significant improvements to OCR quality

### Removed
- Removed terraform directory

### Fixed
- Fixed bug in saving output from the python client


## [1.1.0] - 2025-01-29

### Added
- Added `chunk_processing` config to control chunking
- Added `high_resolution` config to control image density
- Added `segmentation_processing` config to control LLM processing on the segments
- Added `segmentation_strategy` to control segmentation
- Added `expires_in` to API and self deployment config, it is the number of seconds before the task expires and is deleted
- Concurrent OCR and segmentation
- Concurrent page processing
- CPU support - run with `docker compose up -f compose-cpu.yaml -d`
- Python client - `pip install chunkr-ai`
- PATCH `/task/{task_id}` - allows you to update the configuration for a task. Only the steps that are updated will be re-run.
- DELETE `/task/{task_id}` - allows you to delete a task as long as it Status is not `Processing`
- GET `/task/{task_id}/cancel` - allows you to cancel a task before Status is `Processing`
- Helm chart
- Cloudflared tunnel support for https
- Azure support for self deployment
- Minio support for storage
- Python client
- Optionally get base64 encoded files from the API rather than a presigned URL
- Upload base64 encoded files and presigned URLs, when using the Python client

### Changed
- Combined all workers into a `task` worker. See [279](https://github.com/lumina-ai-inc/chunkr/issues/279)
- Redis is now part of the kubernetes deployment
- Documentation
- Improved segmentation quality and speed
- Dashboard has table view - search, deletion, cancellation
- Viewer - better ux
- Better usage tracking - includes graph
- Landing page

### Fixed
- List items incorrect heuristics. See [276](https://github.com/lumina-ai-inc/chunkr/issues/276)
- Reading order

### Removed
(All changes maintain compatibility with old configs)
- Deprecated `model` config
- Deprecated `target_chunk_length`, you can now use `chunk_processing.target_length` instead 
- Deprecated `structured_extraction.json_schema.type`
- Deprecated `ocr_strategy.Off`
- Deprecated `expires_at` in the Python client
