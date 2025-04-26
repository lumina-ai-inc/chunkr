# Changelog

## [1.10.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.9.0...chunkr-core-v1.10.0) (2025-04-26)


### Features

* Fixed reading of text layer (prevent whitespace in between words) ([#461](https://github.com/lumina-ai-inc/chunkr/issues/461)) ([8eba7d3](https://github.com/lumina-ai-inc/chunkr/commit/8eba7d36d108c736f0d0ca658cf90716c2c0c544))

## [1.9.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.8.0...chunkr-core-v1.9.0) (2025-04-23)


### Features

* Update web to use `/task/parse` API route and added `error_handling`, `llm_processing`, `embed_sources` and tokenization settings in the upload component ([#450](https://github.com/lumina-ai-inc/chunkr/issues/450)) ([b1a6aef](https://github.com/lumina-ai-inc/chunkr/commit/b1a6aef41ff8d73daa9ba435e37219b98c765524))

## [1.8.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.7.0...chunkr-core-v1.8.0) (2025-04-15)


### Features

* Allow users to choose an LLM model to use for `segment_processing` by setting the `llm_processing.model_id` param in the POST and PATCH request. The available models can be configured using the `models.yaml` file. ([#437](https://github.com/lumina-ai-inc/chunkr/issues/437)) ([ea526c4](https://github.com/lumina-ai-inc/chunkr/commit/ea526c4c48692ae5d8a9ba00b70008ce238a4c14))

## [1.7.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.6.0...chunkr-core-v1.7.0) (2025-04-06)


### Features

* Added configuration for `error_handling` which allows you to choose between `Fail` or `Continue` on non-critical errors ([0baca0a](https://github.com/lumina-ai-inc/chunkr/commit/0baca0a519b44d139f64d02bec754f259ed329de))


### Bug Fixes

* Default trait added to chunk processing ([20c6f15](https://github.com/lumina-ai-inc/chunkr/commit/20c6f15bf5ef1a538413147103313e65e1223e47))

## [1.6.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.5.2...chunkr-core-v1.6.0) (2025-03-28)


### Features

* Chunking can now be configured using `embed_sources` in `segment_processing.{segment_type}` configuration and allows the choice of pre-configured tokenizers or any huggingface tokenizer by setting the `tokenizer` field in `chunk_processing` ([#420](https://github.com/lumina-ai-inc/chunkr/issues/420)) ([d88ac64](https://github.com/lumina-ai-inc/chunkr/commit/d88ac646ece3935f1c7fcd028bb6c5df0b7d00d3))


### Bug Fixes

* Updated hashmap to lru for caching ([d868c76](https://github.com/lumina-ai-inc/chunkr/commit/d868c76dd16a6751e3baab43190b81e827e26395))

## [1.5.2](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.5.1...chunkr-core-v1.5.2) (2025-03-27)


### Bug Fixes

* **core:** Handle null started_at values with COALESCE in timeout job ([d068be8](https://github.com/lumina-ai-inc/chunkr/commit/d068be82b972a6cd830234448e4bbfe5ebb5245a))

## [1.5.1](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.5.0...chunkr-core-v1.5.1) (2025-03-27)


### Bug Fixes

* Fixed timeout query ([97950e5](https://github.com/lumina-ai-inc/chunkr/commit/97950e54aaa9c10cc5ce42f75600603c27d73168))

## [1.5.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.4.0...chunkr-core-v1.5.0) (2025-03-27)


### Features

* Added doctr small dockers ([#407](https://github.com/lumina-ai-inc/chunkr/issues/407)) ([9b8a56e](https://github.com/lumina-ai-inc/chunkr/commit/9b8a56e273f39aa15d3001c6f7ccb707900dd584))

## [1.4.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.3.3...chunkr-core-v1.4.0) (2025-03-27)


### Features

* **core:** Remove rrq dependency and improve memory management ([92b70dc](https://github.com/lumina-ai-inc/chunkr/commit/92b70dceb1188cec926e415ff295127a3fb085cc))
* New picture prompts ([#405](https://github.com/lumina-ai-inc/chunkr/issues/405)) ([d161fa0](https://github.com/lumina-ai-inc/chunkr/commit/d161fa0820fc03ffaf9bdbbf58c124179548a31a))

## [1.3.3](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.3.2...chunkr-core-v1.3.3) (2025-03-21)


### Bug Fixes

* **core:** Allow PDFs based on extension if the pages can be counted ([#396](https://github.com/lumina-ai-inc/chunkr/issues/396)) ([cfbfd01](https://github.com/lumina-ai-inc/chunkr/commit/cfbfd0155f5fcfb6245acc7dbedb1baa0b12df0b))
* **core:** Auto-fix clippy warnings ([#393](https://github.com/lumina-ai-inc/chunkr/issues/393)) ([0605227](https://github.com/lumina-ai-inc/chunkr/commit/06052278229f0fe1c6feec44172e9048bf09ecc1))
* Fixed prompts and retries for LLMs ([#394](https://github.com/lumina-ai-inc/chunkr/issues/394)) ([4b31588](https://github.com/lumina-ai-inc/chunkr/commit/4b3158889747214abc00ee35c634659491e1c07d))

## [1.3.2](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.3.1...chunkr-core-v1.3.2) (2025-03-20)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#386](https://github.com/lumina-ai-inc/chunkr/issues/386)) ([ccb56f9](https://github.com/lumina-ai-inc/chunkr/commit/ccb56f95212e5840d931893929c6dec648123e34))
* **core:** Update default generation strategies for Picture and Page segments ([5316485](https://github.com/lumina-ai-inc/chunkr/commit/5316485aeec2f923f6fb24f9ab1fcab18e275299))

## [1.3.1](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.3.0...chunkr-core-v1.3.1) (2025-03-16)


### Bug Fixes

* Added retry when finish reason is length ([#383](https://github.com/lumina-ai-inc/chunkr/issues/383)) ([a8dd777](https://github.com/lumina-ai-inc/chunkr/commit/a8dd77791d7294e7166a430776a329e53b0a8103))

## [1.3.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.2.2...chunkr-core-v1.3.0) (2025-03-13)


### Features

* **core:** Added compatibility to Google AI Studio ([#380](https://github.com/lumina-ai-inc/chunkr/issues/380)) ([f56b74c](https://github.com/lumina-ai-inc/chunkr/commit/f56b74c23d1bb0faf050c54a74437139dc9a6938))

## [1.2.2](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.2.1...chunkr-core-v1.2.2) (2025-03-12)


### Features

* **core:** Improved image uploads to pdf conversion and added checkbox support ([a2b65ed](https://github.com/lumina-ai-inc/chunkr/commit/a2b65ed182dcc07af1bccc5b4e98dec3a3335ed8))
