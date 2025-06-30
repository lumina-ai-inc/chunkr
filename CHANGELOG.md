# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.1](https://github.com/lumina-ai-inc/chunkr/compare/v2.1.0...v2.1.1) (2025-06-30)


### Bug Fixes

* Remove keycloakify theme ([ea46cbe](https://github.com/lumina-ai-inc/chunkr/commit/ea46cbea3e6d6abca04f2c79f8c5bd75d496ac7b))

## [2.1.0](https://github.com/lumina-ai-inc/chunkr/compare/v2.0.0...v2.1.0) (2025-06-28)


### Features

* Redesign keycloak auth pages ([adaf74b](https://github.com/lumina-ai-inc/chunkr/commit/adaf74b2c57b37779794e9e3c736905b02bccff3))


### Bug Fixes

* Resolve logger warnings ([#510](https://github.com/lumina-ai-inc/chunkr/issues/510)) ([8810d5e](https://github.com/lumina-ai-inc/chunkr/commit/8810d5ec1ce03c12daa1bee98afed3fb2386cf5a))

## [2.0.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.20.3...v2.0.0) (2025-06-24)


### ⚠ BREAKING CHANGES

* consolidate HTML/markdown generation into single format choice

### Features

* Consolidate HTML/markdown generation into single format choice ([a974f3f](https://github.com/lumina-ai-inc/chunkr/commit/a974f3fbc2bd9158ca052c21a121b479e0eb7613))

## [1.20.3](https://github.com/lumina-ai-inc/chunkr/compare/v1.20.2...v1.20.3) (2025-06-20)


### Bug Fixes

* Updated rust version ([5fce4c4](https://github.com/lumina-ai-inc/chunkr/commit/5fce4c4496dc02954088373b415ba7722ff076be))

## [1.20.2](https://github.com/lumina-ai-inc/chunkr/compare/v1.20.1...v1.20.2) (2025-06-20)


### Bug Fixes

* Reverted landing page hero image ([#548](https://github.com/lumina-ai-inc/chunkr/issues/548)) ([e798336](https://github.com/lumina-ai-inc/chunkr/commit/e7983361fdbb9243c055f2444cacb55aa6072a78))
* Stored cross-site scripting in segmentchunk component ([#546](https://github.com/lumina-ai-inc/chunkr/issues/546)) ([49334b7](https://github.com/lumina-ai-inc/chunkr/commit/49334b788e742f7453c8987e856b57dcb56f0773))

## [1.20.1](https://github.com/lumina-ai-inc/chunkr/compare/v1.20.0...v1.20.1) (2025-05-28)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#541](https://github.com/lumina-ai-inc/chunkr/issues/541)) ([00db663](https://github.com/lumina-ai-inc/chunkr/commit/00db66361022d5566f281badf327a73724da7922))
* Update default for high resolution to true ([#536](https://github.com/lumina-ai-inc/chunkr/issues/536)) ([37cc757](https://github.com/lumina-ai-inc/chunkr/commit/37cc757ea41acce4a662a127bc141e77b56cda03))
* Update robots.txt and blog post page, added sitemap, and removed toast from auth  ([#535](https://github.com/lumina-ai-inc/chunkr/issues/535)) ([3c8f827](https://github.com/lumina-ai-inc/chunkr/commit/3c8f82701d4ff40f932b24607da2dfd394f31e60))
* Updated manual conversion from html to mkd to pandoc ([#539](https://github.com/lumina-ai-inc/chunkr/issues/539)) ([16cc847](https://github.com/lumina-ai-inc/chunkr/commit/16cc847f5728bd963bf8f367579721098190141c))


### Performance Improvements

* Improved llm analytics for extraction errors ([#538](https://github.com/lumina-ai-inc/chunkr/issues/538)) ([0302499](https://github.com/lumina-ai-inc/chunkr/commit/0302499f9942c3f2d0f3888ef419d7e2f6945394))

## [1.20.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.19.0...v1.20.0) (2025-05-28)


### Features

* Added blog pages to frontend - hooked up to Contentful CMS + env toggle for Blog page ([#531](https://github.com/lumina-ai-inc/chunkr/issues/531)) ([90d6dd8](https://github.com/lumina-ai-inc/chunkr/commit/90d6dd88aa5e6cd0bb0580185a6f4fbf3523e35d))

## [1.19.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.18.3...v1.19.0) (2025-05-25)


### Features

* Updated llm processing pipeline to be more robust ([#527](https://github.com/lumina-ai-inc/chunkr/issues/527)) ([1cfb9ca](https://github.com/lumina-ai-inc/chunkr/commit/1cfb9ca10801df57175e5ca148852a48cfea54ed))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#528](https://github.com/lumina-ai-inc/chunkr/issues/528)) ([7a972ed](https://github.com/lumina-ai-inc/chunkr/commit/7a972edddbecce0fcc10edc53669784f6755de89))
* Rm extended context default and picture default ([8411eca](https://github.com/lumina-ai-inc/chunkr/commit/8411eca5dc9c66833cd431049ada51675d3ce81f))

## [1.18.3](https://github.com/lumina-ai-inc/chunkr/compare/v1.18.2...v1.18.3) (2025-05-23)


### Bug Fixes

* Picture default strategy to LLM as default in the generation strategy for html and markdown ([#524](https://github.com/lumina-ai-inc/chunkr/issues/524)) ([e44126c](https://github.com/lumina-ai-inc/chunkr/commit/e44126c0387fb176f9ac6b027e3d6d0231102591))

## [1.18.2](https://github.com/lumina-ai-inc/chunkr/compare/v1.18.1...v1.18.2) (2025-05-22)


### Bug Fixes

* Added file type attributes to span ([#520](https://github.com/lumina-ai-inc/chunkr/issues/520)) ([edd7d0d](https://github.com/lumina-ai-inc/chunkr/commit/edd7d0d140bd3482ce195e6f2243a9e67b4a5efa))
* **core:** Auto-fix clippy warnings ([#518](https://github.com/lumina-ai-inc/chunkr/issues/518)) ([238f47f](https://github.com/lumina-ai-inc/chunkr/commit/238f47fdaf5d2e62d12448424d1018eb1803b8f8))

## [1.18.1](https://github.com/lumina-ai-inc/chunkr/compare/v1.18.0...v1.18.1) (2025-05-22)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#516](https://github.com/lumina-ai-inc/chunkr/issues/516)) ([a938056](https://github.com/lumina-ai-inc/chunkr/commit/a938056779debec5357ad54b27bf5f0788382ba3))
* Improved error handling and telemetry for segment processing ([#515](https://github.com/lumina-ai-inc/chunkr/issues/515)) ([2afc82e](https://github.com/lumina-ai-inc/chunkr/commit/2afc82e361387b51a5d5ab5f99cf74978917e9e1))

## [1.18.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.17.0...v1.18.0) (2025-05-21)


### Features

* **core:** Improved telemetry and added timeout around task processing to deal with long running processes ([#511](https://github.com/lumina-ai-inc/chunkr/issues/511)) ([bbe5913](https://github.com/lumina-ai-inc/chunkr/commit/bbe59130afffedbf5e2e29267afb1f6300918f67))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#512](https://github.com/lumina-ai-inc/chunkr/issues/512)) ([d9ecf60](https://github.com/lumina-ai-inc/chunkr/commit/d9ecf60f308cfe4607673bec172f8fc04d673135))

## [1.17.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.16.0...v1.17.0) (2025-05-20)


### Features

* Added Open telemetry support for better analytics ([#504](https://github.com/lumina-ai-inc/chunkr/issues/504)) ([7baa3d4](https://github.com/lumina-ai-inc/chunkr/commit/7baa3d4a03b5bd15c70dd73b00146adf6dfe7ba6))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#507](https://github.com/lumina-ai-inc/chunkr/issues/507)) ([a8c2e70](https://github.com/lumina-ai-inc/chunkr/commit/a8c2e70fd5db6503fb38273b611fb7ea16c00422))

## [1.16.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.15.1...v1.16.0) (2025-05-17)


### Features

* Added new frontend pdfs in new landing page ([#490](https://github.com/lumina-ai-inc/chunkr/issues/490)) ([bbaf911](https://github.com/lumina-ai-inc/chunkr/commit/bbaf911f205b2f81b723577155e6b5adff246a65))
* Added span class instructions for all formula prompts ([#489](https://github.com/lumina-ai-inc/chunkr/issues/489)) ([5162f8f](https://github.com/lumina-ai-inc/chunkr/commit/5162f8f02fabe8eb0a0f99de1373c5295d3f9ddd))
* Added task level analytics to usage page ([#498](https://github.com/lumina-ai-inc/chunkr/issues/498)) ([e4d63ff](https://github.com/lumina-ai-inc/chunkr/commit/e4d63ffb86c9d790c8bb13cf0cf71642d2f19e2b))
* **core:** Improved error messages on task failure and retry only failed steps ([#496](https://github.com/lumina-ai-inc/chunkr/issues/496)) ([2e09e11](https://github.com/lumina-ai-inc/chunkr/commit/2e09e113f8cf0b0950a77c6954cc9ded2e85c434))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#497](https://github.com/lumina-ai-inc/chunkr/issues/497)) ([8469ddf](https://github.com/lumina-ai-inc/chunkr/commit/8469ddf179709c40949965be203259231bf6b950))

## [1.15.1](https://github.com/lumina-ai-inc/chunkr/compare/v1.15.0...v1.15.1) (2025-05-06)


### Bug Fixes

* Formula prompt selection bug ([#484](https://github.com/lumina-ai-inc/chunkr/issues/484)) ([3b0942b](https://github.com/lumina-ai-inc/chunkr/commit/3b0942b4745739199aa6a8ef11567f05acf4d4cc))

## [1.15.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.14.2...v1.15.0) (2025-05-06)


### Features

* Added extended context (full page + segment) in `segment_processing` ([#480](https://github.com/lumina-ai-inc/chunkr/issues/480)) ([542377b](https://github.com/lumina-ai-inc/chunkr/commit/542377b904aef5fb215bdea3f837315a23eb37de))


### Bug Fixes

* Add Default trait implementation to FallbackStrategy enum ([#479](https://github.com/lumina-ai-inc/chunkr/issues/479)) ([d9a2eaf](https://github.com/lumina-ai-inc/chunkr/commit/d9a2eaf86470d82e8dfed9af874d3cc49ca76ba5))

## [1.14.2](https://github.com/lumina-ai-inc/chunkr/compare/v1.14.1...v1.14.2) (2025-05-01)


### Bug Fixes

* List style points visible now ([#476](https://github.com/lumina-ai-inc/chunkr/issues/476)) ([95a9844](https://github.com/lumina-ai-inc/chunkr/commit/95a98449bf6b6c8f0befd728ceb8206656966b8d))

## [1.14.1](https://github.com/lumina-ai-inc/chunkr/compare/v1.14.0...v1.14.1) (2025-04-30)


### Bug Fixes

* Solved latex rendering errors and final viewer optimizations in place ([#472](https://github.com/lumina-ai-inc/chunkr/issues/472)) ([1fd05c4](https://github.com/lumina-ai-inc/chunkr/commit/1fd05c4ca3b499ddeb7549dbf03988a4e30ea1a8))

## [1.14.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.13.0...v1.14.0) (2025-04-29)


### Features

* Added task retries, started at time, border boxes and more UI enhancements for special segment types like tables, formulas and image descriptions, modified self-host tier based usage section and always showing overage ticker ([#460](https://github.com/lumina-ai-inc/chunkr/issues/460)) ([40a4a69](https://github.com/lumina-ai-inc/chunkr/commit/40a4a6987f82fa01a5cadcbedeee4264bcdb7916))


### Bug Fixes

* Compose-cpu web server replicas ([#467](https://github.com/lumina-ai-inc/chunkr/issues/467)) ([41b2433](https://github.com/lumina-ai-inc/chunkr/commit/41b2433e6ce747c21a0e157e3759b7f1a27ec5ad))
* Updated problematic ML dependencies ([#468](https://github.com/lumina-ai-inc/chunkr/issues/468)) ([5e60961](https://github.com/lumina-ai-inc/chunkr/commit/5e6096122d333b832c8fff1437cb47f70979683e))
* Viewer optimizations and momentum scroll fix ([#466](https://github.com/lumina-ai-inc/chunkr/issues/466)) ([671f840](https://github.com/lumina-ai-inc/chunkr/commit/671f84083eb796b9a120e3ad3f57c7a61cbfcde3))

## [1.13.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.12.1...v1.13.0) (2025-04-26)


### Features

* Fixed reading of text layer (prevent whitespace in between words) ([#461](https://github.com/lumina-ai-inc/chunkr/issues/461)) ([8eba7d3](https://github.com/lumina-ai-inc/chunkr/commit/8eba7d36d108c736f0d0ca658cf90716c2c0c544))

## [1.12.1](https://github.com/lumina-ai-inc/chunkr/compare/v1.12.0...v1.12.1) (2025-04-23)


### Bug Fixes

* Llm models are now fetched from `VITE_API_URL` ([#457](https://github.com/lumina-ai-inc/chunkr/issues/457)) ([06ce1ea](https://github.com/lumina-ai-inc/chunkr/commit/06ce1eaa98048753fd065ddb00908b54914f4857))

## [1.12.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.11.2...v1.12.0) (2025-04-23)


### Features

* Update web to use `/task/parse` API route and added `error_handling`, `llm_processing`, `embed_sources` and tokenization settings in the upload component ([#450](https://github.com/lumina-ai-inc/chunkr/issues/450)) ([b1a6aef](https://github.com/lumina-ai-inc/chunkr/commit/b1a6aef41ff8d73daa9ba435e37219b98c765524))

## [1.11.2](https://github.com/lumina-ai-inc/chunkr/compare/v1.11.1...v1.11.2) (2025-04-22)


### Bug Fixes

* Improved handling of base64, bytes-like and file-like file content in the python client ([#452](https://github.com/lumina-ai-inc/chunkr/issues/452)) ([65e479f](https://github.com/lumina-ai-inc/chunkr/commit/65e479f75ecb91e676afcffe1843d4902a8736e7))

## [1.11.1](https://github.com/lumina-ai-inc/chunkr/compare/v1.11.0...v1.11.1) (2025-04-18)


### Bug Fixes

* Config map in helm chart ([5a08085](https://github.com/lumina-ai-inc/chunkr/commit/5a08085e3c72647dd0833cdcc5574e99948298d5))
* Paths decodable as base64 string can be used with the python client ([#444](https://github.com/lumina-ai-inc/chunkr/issues/444)) ([d544aac](https://github.com/lumina-ai-inc/chunkr/commit/d544aac952d7a6b45ece09b691ad0d1d4b9454c1))

## [1.11.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.10.0...v1.11.0) (2025-04-15)


### Features

* Allow users to choose an LLM model to use for `segment_processing` by setting the `llm_processing.model_id` param in the POST and PATCH request. The available models can be configured using the `models.yaml` file. ([#437](https://github.com/lumina-ai-inc/chunkr/issues/437)) ([ea526c4](https://github.com/lumina-ai-inc/chunkr/commit/ea526c4c48692ae5d8a9ba00b70008ce238a4c14))
* Https support for docker compose using nginx ([#434](https://github.com/lumina-ai-inc/chunkr/issues/434)) ([868096c](https://github.com/lumina-ai-inc/chunkr/commit/868096c8adbb051243e433b2b1c7f440ac1b5997))

## [1.10.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.9.0...v1.10.0) (2025-04-06)


### Features

* Added configuration for `error_handling` which allows you to choose between `Fail` or `Continue` on non-critical errors ([0baca0a](https://github.com/lumina-ai-inc/chunkr/commit/0baca0a519b44d139f64d02bec754f259ed329de))


### Bug Fixes

* Default trait added to chunk processing ([20c6f15](https://github.com/lumina-ai-inc/chunkr/commit/20c6f15bf5ef1a538413147103313e65e1223e47))

## [1.9.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.8.2...v1.9.0) (2025-03-28)


### Features

* Chunking can now be configured using `embed_sources` in `segment_processing.{segment_type}` configuration and allows the choice of pre-configured tokenizers or any huggingface tokenizer by setting the `tokenizer` field in `chunk_processing` ([#420](https://github.com/lumina-ai-inc/chunkr/issues/420)) ([d88ac64](https://github.com/lumina-ai-inc/chunkr/commit/d88ac646ece3935f1c7fcd028bb6c5df0b7d00d3))


### Bug Fixes

* Updated hashmap to lru for caching ([d868c76](https://github.com/lumina-ai-inc/chunkr/commit/d868c76dd16a6751e3baab43190b81e827e26395))

## [1.8.2](https://github.com/lumina-ai-inc/chunkr/compare/v1.8.1...v1.8.2) (2025-03-27)


### Bug Fixes

* **core:** Handle null started_at values with COALESCE in timeout job ([d068be8](https://github.com/lumina-ai-inc/chunkr/commit/d068be82b972a6cd830234448e4bbfe5ebb5245a))

## [1.8.1](https://github.com/lumina-ai-inc/chunkr/compare/v1.8.0...v1.8.1) (2025-03-27)


### Bug Fixes

* Fixed timeout query ([97950e5](https://github.com/lumina-ai-inc/chunkr/commit/97950e54aaa9c10cc5ce42f75600603c27d73168))

## [1.8.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.7.0...v1.8.0) (2025-03-27)


### Features

* Added doctr small dockers ([#407](https://github.com/lumina-ai-inc/chunkr/issues/407)) ([9b8a56e](https://github.com/lumina-ai-inc/chunkr/commit/9b8a56e273f39aa15d3001c6f7ccb707900dd584))

## [1.7.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.6.1...v1.7.0) (2025-03-27)


### Features

* **core:** Remove rrq dependency and improve memory management ([92b70dc](https://github.com/lumina-ai-inc/chunkr/commit/92b70dceb1188cec926e415ff295127a3fb085cc))
* New picture prompts ([#405](https://github.com/lumina-ai-inc/chunkr/issues/405)) ([d161fa0](https://github.com/lumina-ai-inc/chunkr/commit/d161fa0820fc03ffaf9bdbbf58c124179548a31a))

## [1.6.1](https://github.com/lumina-ai-inc/chunkr/compare/v1.6.0...v1.6.1) (2025-03-21)


### Bug Fixes

* **client:** Polling would error out on httpx.ReadTimeout ([#400](https://github.com/lumina-ai-inc/chunkr/issues/400)) ([aea1255](https://github.com/lumina-ai-inc/chunkr/commit/aea125533063de8bbddb36741aed5c1c07ba693b))
* **core:** Allow PDFs based on extension if the pages can be counted ([#396](https://github.com/lumina-ai-inc/chunkr/issues/396)) ([cfbfd01](https://github.com/lumina-ai-inc/chunkr/commit/cfbfd0155f5fcfb6245acc7dbedb1baa0b12df0b))
* **core:** Auto-fix clippy warnings ([#393](https://github.com/lumina-ai-inc/chunkr/issues/393)) ([0605227](https://github.com/lumina-ai-inc/chunkr/commit/06052278229f0fe1c6feec44172e9048bf09ecc1))
* Fixed prompts and retries for LLMs ([#394](https://github.com/lumina-ai-inc/chunkr/issues/394)) ([4b31588](https://github.com/lumina-ai-inc/chunkr/commit/4b3158889747214abc00ee35c634659491e1c07d))

## [1.6.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.5.1...v1.6.0) (2025-03-20)


### Features

* Added new cropped image viewing, updated upload component defaults for image VLM processing, and some bug fixes for segment highlighting + JSON viewing ([#388](https://github.com/lumina-ai-inc/chunkr/issues/388)) ([6115ee0](https://github.com/lumina-ai-inc/chunkr/commit/6115ee08b785e94ed8432e4c75da98e32a42bea9))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#386](https://github.com/lumina-ai-inc/chunkr/issues/386)) ([ccb56f9](https://github.com/lumina-ai-inc/chunkr/commit/ccb56f95212e5840d931893929c6dec648123e34))
* **core:** Update default generation strategies for Picture and Page segments ([5316485](https://github.com/lumina-ai-inc/chunkr/commit/5316485aeec2f923f6fb24f9ab1fcab18e275299))
* Downgraded cuda version for doctr ([36db353](https://github.com/lumina-ai-inc/chunkr/commit/36db353079aaf56fd4613ea13b3c88e7d678e897))

## [1.5.1](https://github.com/lumina-ai-inc/chunkr/compare/v1.5.0...v1.5.1) (2025-03-16)


### Bug Fixes

* Added imagemagick to docker images ([d3ac921](https://github.com/lumina-ai-inc/chunkr/commit/d3ac9215f0c570269ba16f3855512da606fd3d4c))
* Added retry when finish reason is length ([#383](https://github.com/lumina-ai-inc/chunkr/issues/383)) ([a8dd777](https://github.com/lumina-ai-inc/chunkr/commit/a8dd77791d7294e7166a430776a329e53b0a8103))
* Correct Rust lint workflow configuration ([0b1a1eb](https://github.com/lumina-ai-inc/chunkr/commit/0b1a1ebdf42a2c22ddfcff52fb7356ebb4216287))

## [1.5.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.4.2...v1.5.0) (2025-03-13)


### Features

* **core:** Added compatibility to Google AI Studio ([#380](https://github.com/lumina-ai-inc/chunkr/issues/380)) ([f56b74c](https://github.com/lumina-ai-inc/chunkr/commit/f56b74c23d1bb0faf050c54a74437139dc9a6938))


### Bug Fixes

* Fix keycloak tag ([df9efa5](https://github.com/lumina-ai-inc/chunkr/commit/df9efa5e212a517020e47d66c3820e62ca87acf2))

## [1.4.2](https://github.com/lumina-ai-inc/chunkr/compare/v1.4.1...v1.4.2) (2025-03-12)


### Bug Fixes

* Github action now removes v from version before tagging ([6c77a1f](https://github.com/lumina-ai-inc/chunkr/commit/6c77a1f5f435c362ec62aabb8bd29a78cc7eba1e))
* Moved infrastructure from values.yaml to infrastructure.yaml ([e4ba284](https://github.com/lumina-ai-inc/chunkr/commit/e4ba284b85c3290f585abce36d97c8c9860bdb9a))

## [1.4.1](https://github.com/lumina-ai-inc/chunkr/compare/v1.4.0...v1.4.1) (2025-03-12)


### Bug Fixes

* Continue on error on docker build ([aca0b44](https://github.com/lumina-ai-inc/chunkr/commit/aca0b4444875a1b053924a60380e6ee44a4dc005))

## [1.4.0](https://github.com/lumina-ai-inc/chunkr/compare/v1.3.5...v1.4.0) (2025-03-12)


### Features

* /health return current version ([627e8c9](https://github.com/lumina-ai-inc/chunkr/commit/627e8c9a1160bf4a360f6d0ea0f1376f64344642))


### Bug Fixes

* Updated changelog paths ([d20b811](https://github.com/lumina-ai-inc/chunkr/commit/d20b8112fc5043f5eecabf1e72e89412b1b5e7b1))

## [1.3.5](https://github.com/lumina-ai-inc/chunkr/compare/v1.3.4...v1.3.5) (2025-03-12)


### Bug Fixes

* Added back segmentation docker with self hosted runner ([0984ba2](https://github.com/lumina-ai-inc/chunkr/commit/0984ba2710fca19a807985e5a92fbf1e185bbb03))

## [1.3.4](https://github.com/lumina-ai-inc/chunkr/compare/v1.3.3...v1.3.4) (2025-03-11)


### Bug Fixes

* Removed segmenetation from docker build ([5dc9e6e](https://github.com/lumina-ai-inc/chunkr/commit/5dc9e6e5d1687bbe6ab3555f7df5656856a43f34))

## [1.3.3](https://github.com/lumina-ai-inc/chunkr/compare/v1.3.2...v1.3.3) (2025-03-11)


### Bug Fixes

* Updated rust version for docker builds ([e5a3633](https://github.com/lumina-ai-inc/chunkr/commit/e5a3633e970dacae3ce08e42f5d7249aed592fa6))

## [1.3.2](https://github.com/lumina-ai-inc/chunkr/compare/v1.3.1...v1.3.2) (2025-03-11)


### Bug Fixes

* Release-please docker build ([6e1ff43](https://github.com/lumina-ai-inc/chunkr/commit/6e1ff43ad0d5780d2f4a6e67b0b2bcc47d8964f6))

## [1.3.1](https://github.com/lumina-ai-inc/chunkr/compare/v1.3.0...v1.3.1) (2025-03-11)


### Bug Fixes

* Docker compose updated uses pr ([f45abd1](https://github.com/lumina-ai-inc/chunkr/commit/f45abd130d4c643c288c3492bb27f6736059dfbf))

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
