# Changelog

## [1.15.1](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.15.0...chunkr-core-v1.15.1) (2025-05-28)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#541](https://github.com/lumina-ai-inc/chunkr/issues/541)) ([00db663](https://github.com/lumina-ai-inc/chunkr/commit/00db66361022d5566f281badf327a73724da7922))
* Update default for high resolution to true ([#536](https://github.com/lumina-ai-inc/chunkr/issues/536)) ([37cc757](https://github.com/lumina-ai-inc/chunkr/commit/37cc757ea41acce4a662a127bc141e77b56cda03))
* Updated manual conversion from html to mkd to pandoc ([#539](https://github.com/lumina-ai-inc/chunkr/issues/539)) ([16cc847](https://github.com/lumina-ai-inc/chunkr/commit/16cc847f5728bd963bf8f367579721098190141c))


### Performance Improvements

* Improved llm analytics for extraction errors ([#538](https://github.com/lumina-ai-inc/chunkr/issues/538)) ([0302499](https://github.com/lumina-ai-inc/chunkr/commit/0302499f9942c3f2d0f3888ef419d7e2f6945394))

## [1.15.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.14.3...chunkr-core-v1.15.0) (2025-05-25)


### Features

* Updated llm processing pipeline to be more robust ([#527](https://github.com/lumina-ai-inc/chunkr/issues/527)) ([1cfb9ca](https://github.com/lumina-ai-inc/chunkr/commit/1cfb9ca10801df57175e5ca148852a48cfea54ed))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#528](https://github.com/lumina-ai-inc/chunkr/issues/528)) ([7a972ed](https://github.com/lumina-ai-inc/chunkr/commit/7a972edddbecce0fcc10edc53669784f6755de89))
* Rm extended context default and picture default ([8411eca](https://github.com/lumina-ai-inc/chunkr/commit/8411eca5dc9c66833cd431049ada51675d3ce81f))

## [1.14.3](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.14.2...chunkr-core-v1.14.3) (2025-05-23)


### Bug Fixes

* Picture default strategy to LLM as default in the generation strategy for html and markdown ([#524](https://github.com/lumina-ai-inc/chunkr/issues/524)) ([e44126c](https://github.com/lumina-ai-inc/chunkr/commit/e44126c0387fb176f9ac6b027e3d6d0231102591))

## [1.14.2](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.14.1...chunkr-core-v1.14.2) (2025-05-22)


### Bug Fixes

* Added file type attributes to span ([#520](https://github.com/lumina-ai-inc/chunkr/issues/520)) ([edd7d0d](https://github.com/lumina-ai-inc/chunkr/commit/edd7d0d140bd3482ce195e6f2243a9e67b4a5efa))
* **core:** Auto-fix clippy warnings ([#518](https://github.com/lumina-ai-inc/chunkr/issues/518)) ([238f47f](https://github.com/lumina-ai-inc/chunkr/commit/238f47fdaf5d2e62d12448424d1018eb1803b8f8))

## [1.14.1](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.14.0...chunkr-core-v1.14.1) (2025-05-22)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#516](https://github.com/lumina-ai-inc/chunkr/issues/516)) ([a938056](https://github.com/lumina-ai-inc/chunkr/commit/a938056779debec5357ad54b27bf5f0788382ba3))
* Improved error handling and telemetry for segment processing ([#515](https://github.com/lumina-ai-inc/chunkr/issues/515)) ([2afc82e](https://github.com/lumina-ai-inc/chunkr/commit/2afc82e361387b51a5d5ab5f99cf74978917e9e1))

## [1.14.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.13.0...chunkr-core-v1.14.0) (2025-05-21)


### Features

* **core:** Improved telemetry and added timeout around task processing to deal with long running processes ([#511](https://github.com/lumina-ai-inc/chunkr/issues/511)) ([bbe5913](https://github.com/lumina-ai-inc/chunkr/commit/bbe59130afffedbf5e2e29267afb1f6300918f67))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#512](https://github.com/lumina-ai-inc/chunkr/issues/512)) ([d9ecf60](https://github.com/lumina-ai-inc/chunkr/commit/d9ecf60f308cfe4607673bec172f8fc04d673135))

## [1.13.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.12.0...chunkr-core-v1.13.0) (2025-05-20)


### Features

* Added Open telemetry support for better analytics ([#504](https://github.com/lumina-ai-inc/chunkr/issues/504)) ([7baa3d4](https://github.com/lumina-ai-inc/chunkr/commit/7baa3d4a03b5bd15c70dd73b00146adf6dfe7ba6))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#507](https://github.com/lumina-ai-inc/chunkr/issues/507)) ([a8c2e70](https://github.com/lumina-ai-inc/chunkr/commit/a8c2e70fd5db6503fb38273b611fb7ea16c00422))

## [1.12.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.11.1...chunkr-core-v1.12.0) (2025-05-17)


### Features

* Added span class instructions for all formula prompts ([#489](https://github.com/lumina-ai-inc/chunkr/issues/489)) ([5162f8f](https://github.com/lumina-ai-inc/chunkr/commit/5162f8f02fabe8eb0a0f99de1373c5295d3f9ddd))
* **core:** Improved error messages on task failure and retry only failed steps ([#496](https://github.com/lumina-ai-inc/chunkr/issues/496)) ([2e09e11](https://github.com/lumina-ai-inc/chunkr/commit/2e09e113f8cf0b0950a77c6954cc9ded2e85c434))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#497](https://github.com/lumina-ai-inc/chunkr/issues/497)) ([8469ddf](https://github.com/lumina-ai-inc/chunkr/commit/8469ddf179709c40949965be203259231bf6b950))

## [1.11.1](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.11.0...chunkr-core-v1.11.1) (2025-05-06)


### Bug Fixes

* Formula prompt selection bug ([#484](https://github.com/lumina-ai-inc/chunkr/issues/484)) ([3b0942b](https://github.com/lumina-ai-inc/chunkr/commit/3b0942b4745739199aa6a8ef11567f05acf4d4cc))

## [1.11.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-core-v1.10.0...chunkr-core-v1.11.0) (2025-05-06)


### Features

* Added extended context (full page + segment) in `segment_processing` ([#480](https://github.com/lumina-ai-inc/chunkr/issues/480)) ([542377b](https://github.com/lumina-ai-inc/chunkr/commit/542377b904aef5fb215bdea3f837315a23eb37de))


### Bug Fixes

* Add Default trait implementation to FallbackStrategy enum ([#479](https://github.com/lumina-ai-inc/chunkr/issues/479)) ([d9a2eaf](https://github.com/lumina-ai-inc/chunkr/commit/d9a2eaf86470d82e8dfed9af874d3cc49ca76ba5))

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
