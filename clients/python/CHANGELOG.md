# Changelog

## [0.0.45](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-ai-v0.0.44...chunkr-ai-v0.0.45) (2025-04-06)


### Features

* Added configuration for `error_handling` which allows you to choose between `Fail` or `Continue` on non-critical errors ([0baca0a](https://github.com/lumina-ai-inc/chunkr/commit/0baca0a519b44d139f64d02bec754f259ed329de))

## [0.0.44](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-ai-v0.0.43...chunkr-ai-v0.0.44) (2025-03-28)


### Features

* Chunking can now be configured using `embed_sources` in `segment_processing.{segment_type}` configuration and allows the choice of pre-configured tokenizers or any huggingface tokenizer by setting the `tokenizer` field in `chunk_processing` ([#420](https://github.com/lumina-ai-inc/chunkr/issues/420)) ([d88ac64](https://github.com/lumina-ai-inc/chunkr/commit/d88ac646ece3935f1c7fcd028bb6c5df0b7d00d3))

## [0.0.43](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-ai-v0.0.42...chunkr-ai-v0.0.43) (2025-03-21)


### Bug Fixes

* **client:** Polling would error out on httpx.ReadTimeout ([#400](https://github.com/lumina-ai-inc/chunkr/issues/400)) ([aea1255](https://github.com/lumina-ai-inc/chunkr/commit/aea125533063de8bbddb36741aed5c1c07ba693b))

## [0.0.42](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-ai-v0.0.41...chunkr-ai-v0.0.42) (2025-03-11)


### Bug Fixes

* Await was missing in response ([1ad37d8](https://github.com/lumina-ai-inc/chunkr/commit/1ad37d851ee0379c13ba663fc8bafb3541e409a2))
* Await was missing in response ([632adce](https://github.com/lumina-ai-inc/chunkr/commit/632adce42c7850a788e0e46817e2498724c76890))
