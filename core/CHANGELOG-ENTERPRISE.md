# Changelog

## [2.3.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v2.2.0-enterprise...chunkr-core-enterprise-v2.3.0-enterprise) (2025-07-07)


### Features

* Added vertex ai support ([#143](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/143)) ([2ce9327](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/2ce9327a2c02bd4d0a3ca22fc67c24119ceddc28))

## [2.2.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v2.1.0-enterprise...chunkr-core-enterprise-v2.2.0-enterprise) (2025-07-04)


### Features

* Enhance excel parser to identify all element types with improved error handling ([#133](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/133)) ([9f15fd7](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/9f15fd73b39bd529aba7af245822deeb4dce7cc6))


### Bug Fixes

* Browser errors on kube ([#128](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/128)) ([418af50](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/418af50cffbc34ece97d4de0e51b56ac2bcfdeb5))

## [2.1.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v2.0.0-enterprise...chunkr-core-enterprise-v2.1.0-enterprise) (2025-07-03)


### Features

* Added flag for experimental features, and set default for spreadsheet feature to be off ([455a13e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/455a13ebd05f2a15fa68a2bdefda38d6b367a50e))


### Bug Fixes

* Added chrome sandbox false for docker containers ([181debd](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/181debd637e051610f5ed0e9afd9ae5e8581ba5a))

## [2.0.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.16.1-enterprise...chunkr-core-enterprise-v2.0.0-enterprise) (2025-06-29)


### ⚠ BREAKING CHANGES

* consolidate HTML/markdown generation into single format choice

### Features

* Add advanced Excel parser with automatic table identification, LibreOffice-based HTML conversion, headless Chrome rendering, and specialized spreadsheet chunking pipeline ([5616c21](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5616c2107f35bdcbd6b6933c878043dc53665611))
* Consolidate HTML/markdown generation into single format choice ([a974f3f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a974f3fbc2bd9158ca052c21a121b479e0eb7613))


### Bug Fixes

* Added timeouts to azure and improved error messages ([#119](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/119)) ([f8bb3f1](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/f8bb3f1ae31fd668a34177900a07bd0b8dd09241))

## [1.16.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.16.0-enterprise...chunkr-core-enterprise-v1.16.1-enterprise) (2025-05-29)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#541](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/541)) ([00db663](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/00db66361022d5566f281badf327a73724da7922))
* Update default for high resolution to true ([#536](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/536)) ([37cc757](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/37cc757ea41acce4a662a127bc141e77b56cda03))
* Updated manual conversion from html to mkd to pandoc ([#539](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/539)) ([16cc847](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/16cc847f5728bd963bf8f367579721098190141c))


### Performance Improvements

* Improved llm analytics for extraction errors ([#538](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/538)) ([0302499](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/0302499f9942c3f2d0f3888ef419d7e2f6945394))

## [1.16.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.15.4-enterprise...chunkr-core-enterprise-v1.16.0-enterprise) (2025-05-28)


### Features

* Updated llm processing pipeline to be more robust ([#527](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/527)) ([1cfb9ca](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/1cfb9ca10801df57175e5ca148852a48cfea54ed))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#528](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/528)) ([7a972ed](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/7a972edddbecce0fcc10edc53669784f6755de89))

## [1.15.4-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.15.3-enterprise...chunkr-core-enterprise-v1.15.4-enterprise) (2025-05-24)


### Bug Fixes

* Rm extended context default and picture default ([3772841](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/37728415a99290f57bbc22d8f62bdf025b22adcb))
* Rm extended context default and picture default ([8411eca](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/8411eca5dc9c66833cd431049ada51675d3ce81f))

## [1.15.3-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.15.2-enterprise...chunkr-core-enterprise-v1.15.3-enterprise) (2025-05-23)


### Bug Fixes

* Picture default strategy to LLM as default in the generation strategy for html and markdown ([#524](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/524)) ([e44126c](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e44126c0387fb176f9ac6b027e3d6d0231102591))

## [1.15.2-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.15.1-enterprise...chunkr-core-enterprise-v1.15.2-enterprise) (2025-05-22)


### Bug Fixes

* Added file type attributes to span ([#520](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/520)) ([edd7d0d](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/edd7d0d140bd3482ce195e6f2243a9e67b4a5efa))

## [1.15.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.15.0-enterprise...chunkr-core-enterprise-v1.15.1-enterprise) (2025-05-22)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#516](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/516)) ([a938056](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a938056779debec5357ad54b27bf5f0788382ba3))
* **core:** Auto-fix clippy warnings ([#518](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/518)) ([238f47f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/238f47fdaf5d2e62d12448424d1018eb1803b8f8))
* Improved error handling and telemetry for segment processing ([#515](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/515)) ([2afc82e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/2afc82e361387b51a5d5ab5f99cf74978917e9e1))

## [1.15.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.14.0-enterprise...chunkr-core-enterprise-v1.15.0-enterprise) (2025-05-22)


### Features

* **core:** Improved telemetry and added timeout around task processing to deal with long running processes ([#511](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/511)) ([bbe5913](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/bbe59130afffedbf5e2e29267afb1f6300918f67))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#512](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/512)) ([d9ecf60](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d9ecf60f308cfe4607673bec172f8fc04d673135))

## [1.14.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.13.0-enterprise...chunkr-core-enterprise-v1.14.0-enterprise) (2025-05-20)


### Features

* Added Open telemetry support for better analytics ([#504](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/504)) ([7baa3d4](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/7baa3d4a03b5bd15c70dd73b00146adf6dfe7ba6))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#507](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/507)) ([a8c2e70](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a8c2e70fd5db6503fb38273b611fb7ea16c00422))

## [1.13.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.12.2-enterprise...chunkr-core-enterprise-v1.13.0-enterprise) (2025-05-17)


### Features

* Added span class instructions for all formula prompts ([#489](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/489)) ([5162f8f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5162f8f02fabe8eb0a0f99de1373c5295d3f9ddd))
* **core:** Improved error messages on task failure and retry only failed steps ([#496](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/496)) ([2e09e11](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/2e09e113f8cf0b0950a77c6954cc9ded2e85c434))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#497](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/497)) ([8469ddf](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/8469ddf179709c40949965be203259231bf6b950))

## [1.12.2-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.12.1-enterprise...chunkr-core-enterprise-v1.12.2-enterprise) (2025-05-12)


### Bug Fixes

* Address Clippy warnings ([5fee794](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5fee794ef7bc3cc9fcc6b684a839c67281d7e566))

## [1.12.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-core-enterprise-v1.12.0-enterprise...chunkr-core-enterprise-v1.12.1-enterprise) (2025-05-06)


### Bug Fixes

* Formula prompt selection bug ([#484](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/484)) ([1762c26](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/1762c264b1c8e8989d51ebfada461fe85c7b6e10))

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
