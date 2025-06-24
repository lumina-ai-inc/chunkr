# Changelog

## [0.1.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-ai-v0.0.50...chunkr-ai-v0.1.0) (2025-06-24)


### ⚠ BREAKING CHANGES

* consolidate HTML/markdown generation into single format choice

### Features

* Consolidate HTML/markdown generation into single format choice ([a974f3f](https://github.com/lumina-ai-inc/chunkr/commit/a974f3fbc2bd9158ca052c21a121b479e0eb7613))

## [0.0.50](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-ai-v0.0.49...chunkr-ai-v0.0.50) (2025-05-22)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#518](https://github.com/lumina-ai-inc/chunkr/issues/518)) ([238f47f](https://github.com/lumina-ai-inc/chunkr/commit/238f47fdaf5d2e62d12448424d1018eb1803b8f8))

## [0.0.49](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-ai-v0.0.48...chunkr-ai-v0.0.49) (2025-05-06)


### Features

* Added extended context (full page + segment) in `segment_processing` ([#480](https://github.com/lumina-ai-inc/chunkr/issues/480)) ([542377b](https://github.com/lumina-ai-inc/chunkr/commit/542377b904aef5fb215bdea3f837315a23eb37de))

## [0.0.48](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-ai-v0.0.47...chunkr-ai-v0.0.48) (2025-04-22)


### Bug Fixes

* Improved handling of base64, bytes-like and file-like file content in the python client ([#452](https://github.com/lumina-ai-inc/chunkr/issues/452)) ([65e479f](https://github.com/lumina-ai-inc/chunkr/commit/65e479f75ecb91e676afcffe1843d4902a8736e7))

## [0.0.47](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-ai-v0.0.46...chunkr-ai-v0.0.47) (2025-04-18)


### Bug Fixes

* Paths decodable as base64 string can be used with the python client ([#444](https://github.com/lumina-ai-inc/chunkr/issues/444)) ([d544aac](https://github.com/lumina-ai-inc/chunkr/commit/d544aac952d7a6b45ece09b691ad0d1d4b9454c1))

## [0.0.46](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-ai-v0.0.45...chunkr-ai-v0.0.46) (2025-04-15)


### Features

* Allow users to choose an LLM model to use for `segment_processing` by setting the `llm_processing.model_id` param in the POST and PATCH request. The available models can be configured using the `models.yaml` file. ([#437](https://github.com/lumina-ai-inc/chunkr/issues/437)) ([ea526c4](https://github.com/lumina-ai-inc/chunkr/commit/ea526c4c48692ae5d8a9ba00b70008ce238a4c14))

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
