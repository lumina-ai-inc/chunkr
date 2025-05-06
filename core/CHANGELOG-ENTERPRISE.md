# Changelog

## [1.12.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.11.1-enterprise...chunkr-core-enterprise-v1.12.0-enterprise) (2025-05-06)


### Features

* Added extended context (full page + segment) in `segment_processing` ([#480](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/480)) ([542377b](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/542377b904aef5fb215bdea3f837315a23eb37de))


### Bug Fixes

* Add Default trait implementation to FallbackStrategy enum ([#479](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/479)) ([d9a2eaf](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d9a2eaf86470d82e8dfed9af874d3cc49ca76ba5))

## [1.11.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.11.0-enterprise...chunkr-core-enterprise-v1.11.1-enterprise) (2025-05-01)


### Bug Fixes

* New mu trigger ([#69](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/69)) ([7a6993e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/7a6993e1a36860e238a38626b12c7350be909afa))

## [1.11.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.10.0-enterprise...chunkr-core-enterprise-v1.11.0-enterprise) (2025-04-26)


### Features

* Fixed reading of text layer (prevent whitespace in between words) ([#461](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/461)) ([8eba7d3](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/8eba7d36d108c736f0d0ca658cf90716c2c0c544))

## [1.10.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.9.0-enterprise...chunkr-core-enterprise-v1.10.0-enterprise) (2025-04-23)


### Features

* Saved myself from disaster ([eba8ce4](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/eba8ce4c6f46e0a00dc86e4bc5bad664328b8630))
* Update web to use `/task/parse` API route and added `error_handling`, `llm_processing`, `embed_sources`, and tokenization settings in the upload component ([#450](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/450)) ([13ec8c7](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/13ec8c772ecdb54983fd009be7e59e37b3695ba1))

## [1.9.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.8.0-enterprise...chunkr-core-enterprise-v1.9.0-enterprise) (2025-04-16)


### Features

* Allow users to choose an LLM model to use for `segment_processing` by setting the `llm_processing.model_id` param in the POST and PATCH request. The available models can be configured using the `models.yaml` file. ([#437](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/437)) ([ea526c4](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/ea526c4c48692ae5d8a9ba00b70008ce238a4c14))

## [1.8.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.7.0-enterprise...chunkr-core-enterprise-v1.8.0-enterprise) (2025-04-07)


### Features

* Added configuration for `error_handling` which allows you to choose between `Fail` or `Continue` on non-critical errors ([0baca0a](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/0baca0a519b44d139f64d02bec754f259ed329de))


### Bug Fixes

* Default trait added to chunk processing ([20c6f15](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/20c6f15bf5ef1a538413147103313e65e1223e47))

## [1.7.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.6.0-enterprise...chunkr-core-enterprise-v1.7.0-enterprise) (2025-03-29)


### Features

* Chunking can now be configured using `embed_sources` in `segment_processing.{segment_type}` configuration and allows the choice of pre-configured tokenizers or any huggingface tokenizer by setting the `tokenizer` field in `chunk_processing` ([#420](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/420)) ([d88ac64](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d88ac646ece3935f1c7fcd028bb6c5df0b7d00d3))


### Bug Fixes

* Updated hashmap to lru for caching ([d868c76](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d868c76dd16a6751e3baab43190b81e827e26395))

## [1.6.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.5.0-enterprise...chunkr-core-enterprise-v1.6.0-enterprise) (2025-03-27)


### Features

* Added doctr small dockers ([#407](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/407)) ([9b8a56e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/9b8a56e273f39aa15d3001c6f7ccb707900dd584))


### Bug Fixes

* **core:** Handle null started_at values with COALESCE in timeout job ([d068be8](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d068be82b972a6cd830234448e4bbfe5ebb5245a))
* Fixed timeout query ([97950e5](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/97950e54aaa9c10cc5ce42f75600603c27d73168))

## [1.5.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.4.5-enterprise...chunkr-core-enterprise-v1.5.0-enterprise) (2025-03-27)


### Features

* **core:** Remove rrq dependency and improve memory management ([92b70dc](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/92b70dceb1188cec926e415ff295127a3fb085cc))
* New picture prompts ([#405](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/405)) ([d161fa0](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d161fa0820fc03ffaf9bdbbf58c124179548a31a))

## [1.4.5-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.4.4-enterprise...chunkr-core-enterprise-v1.4.5-enterprise) (2025-03-21)


### Bug Fixes

* **core:** Allow PDFs based on extension if the pages can be counted ([#396](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/396)) ([cfbfd01](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/cfbfd0155f5fcfb6245acc7dbedb1baa0b12df0b))
* **core:** Auto-fix clippy warnings ([#393](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/393)) ([0605227](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/06052278229f0fe1c6feec44172e9048bf09ecc1))

## [1.4.4-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.4.3-enterprise...chunkr-core-enterprise-v1.4.4-enterprise) (2025-03-21)


### Bug Fixes

* Fixed prompts and retries for LLMs ([#394](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/394)) ([4b31588](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/4b3158889747214abc00ee35c634659491e1c07d))

## [1.4.3-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.4.2-enterprise...chunkr-core-enterprise-v1.4.3-enterprise) (2025-03-20)


### Bug Fixes

* **core:** Update default generation strategies for Picture and Page segments ([5316485](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5316485aeec2f923f6fb24f9ab1fcab18e275299))

## [1.4.2-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.4.1-enterprise...chunkr-core-enterprise-v1.4.2-enterprise) (2025-03-19)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#28](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/28)) ([9fe6d9f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/9fe6d9f2d3dbbe3cf284f39c077f18c0fc902171))
* **core:** Auto-fix clippy warnings ([#386](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/386)) ([ccb56f9](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/ccb56f95212e5840d931893929c6dec648123e34))

## [1.4.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.4.0-enterprise...chunkr-core-enterprise-v1.4.1-enterprise) (2025-03-14)


### Bug Fixes

* LLM retry when stop reason is length ([e8b3e7c](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e8b3e7cccb65874afbb00cd24ec0524ead9a1f0c))

## [1.4.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.3.0-enterprise...chunkr-core-enterprise-v1.4.0-enterprise) (2025-03-13)


### Features

* **core:** Added compatibility to Google AI Studio ([#380](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/380)) ([f56b74c](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/f56b74c23d1bb0faf050c54a74437139dc9a6938))

## [1.3.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.2.2-enterprise...chunkr-core-enterprise-v1.3.0-enterprise) (2025-03-13)


### Features

* Added release please for automated releases ([#363](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/363)) ([d808d4e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d808d4e72464b83590dfab73fe973e2f98b4f7e7))
* **core:** Improved image uploads to pdf conversion and added checkbox support ([a2b65ed](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a2b65ed182dcc07af1bccc5b4e98dec3a3335ed8))


### Bug Fixes

* Debugging please release with core changes ([558a6f9](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/558a6f9fd86c5d6e53b770dd48909a3a60e7f110))
* Removed changelog from core ([b658a63](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/b658a6373baee8cba156d7272a8c91accda0e0e8))
* Small bugfix ([101c623](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/101c62301994347331382cf33e4e15bfdfae0013))
