# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.2.0...v1.3.0) (2025-03-11)


### Features

* Added automation of docker build ([#365](https://github.com/lumina-ai-inc/chunkr/issues/365)) ([f01cb2f](https://github.com/lumina-ai-inc/chunkr/commit/f01cb2fc66c104066f1188149cdbbb8390337169))


### Bug Fixes

* Debugging please release ([e574177](https://github.com/lumina-ai-inc/chunkr/commit/e574177cc28c68e86ab08ac5b83328b393b02bf4))
* Debugging please release with core changes ([558a6f9](https://github.com/lumina-ai-inc/chunkr/commit/558a6f9fd86c5d6e53b770dd48909a3a60e7f110))
* Docker builds use root version ([82e1768](https://github.com/lumina-ai-inc/chunkr/commit/82e176868e215f550377d9aed91e5b37fd57faba))
* Docker compose files update separately ([15328a2](https://github.com/lumina-ai-inc/chunkr/commit/15328a23dfd4399b6a56babb18becd04bf7bdf72))
* Image tag updates not full image ([7b8791f](https://github.com/lumina-ai-inc/chunkr/commit/7b8791f6bdee1e2b5f47496936700de4ddaee537))
* Only trigger docker build after releases created ([676c280](https://github.com/lumina-ai-inc/chunkr/commit/676c280e975ea37a8a737876854b0e3aa7006fc2))

## [1.2.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.1.0...v1.2.0) (2025-03-11)


### Features

* Added release please for automated releases ([#363](https://github.com/lumina-ai-inc/chunkr/issues/363)) ([d808d4e](https://github.com/lumina-ai-inc/chunkr/commit/d808d4e72464b83590dfab73fe973e2f98b4f7e7))


### Bug Fixes

* Await was missing in response ([1ad37d8](https://github.com/lumina-ai-inc/chunkr/commit/1ad37d851ee0379c13ba663fc8bafb3541e409a2))
* Await was missing in response ([632adce](https://github.com/lumina-ai-inc/chunkr/commit/632adce42c7850a788e0e46817e2498724c76890))

### Added
- Added route `POST /task/parse` and `PATCH /task/{task_id}/parse` to parse a task. These routes are exactly the same as the `POST /task` and `PATCH /task/{task_id}` routes, but don't use a multipart request.
>The old routes are deprecated but will continue to work for the foreseeable future.
- Batch parallelization, so individual tasks can take full advantage of unused GPU resources.

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
