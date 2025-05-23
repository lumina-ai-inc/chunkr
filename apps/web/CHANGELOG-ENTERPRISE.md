# Changelog

## [1.6.3-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-web-enterprise-v1.6.2-enterprise...chunkr-web-enterprise-v1.6.3-enterprise) (2025-05-23)


### Bug Fixes

* Merge remnant ([ef2dbe5](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/ef2dbe57182fcb85c75a30a021feb5393c17d7f2))

## [1.6.2-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-web-enterprise-v1.6.1-enterprise...chunkr-web-enterprise-v1.6.2-enterprise) (2025-05-22)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#518](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/518)) ([238f47f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/238f47fdaf5d2e62d12448424d1018eb1803b8f8))

## [1.6.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-web-enterprise-v1.6.0-enterprise...chunkr-web-enterprise-v1.6.1-enterprise) (2025-05-22)


### Bug Fixes

* Compile errors ([353e79f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/353e79f0b000fb37204a42ba6deedff949ee335d))

## [1.6.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-web-enterprise-v1.5.0-enterprise...chunkr-web-enterprise-v1.6.0-enterprise) (2025-05-17)


### Features

* Added classification pipeline for PDFs (making data for DLA training) ([#86](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/86)) ([e8ca16c](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e8ca16ca9b16fe02df5bfe8b01e831ce60933fde))
* Added new frontend pdfs in new landing page ([#490](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/490)) ([bbaf911](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/bbaf911f205b2f81b723577155e6b5adff246a65))
* Added task level analytics to usage page ([#498](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/498)) ([e4d63ff](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e4d63ffb86c9d790c8bb13cf0cf71642d2f19e2b))

## [1.5.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-web-enterprise-v1.4.2-enterprise...chunkr-web-enterprise-v1.5.0-enterprise) (2025-05-06)


### Features

* Added extended context (full page + segment) in `segment_processing` ([#480](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/480)) ([542377b](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/542377b904aef5fb215bdea3f837315a23eb37de))

## [1.4.2-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-web-enterprise-v1.4.1-enterprise...chunkr-web-enterprise-v1.4.2-enterprise) (2025-05-01)


### Bug Fixes

* New mu trigger ([#69](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/69)) ([7a6993e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/7a6993e1a36860e238a38626b12c7350be909afa))

## [1.4.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-web-enterprise-v1.4.0-enterprise...chunkr-web-enterprise-v1.4.1-enterprise) (2025-05-01)


### Bug Fixes

* Solved latex rendering errors and final viewer optimizations in place ([#472](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/472)) ([1fd05c4](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/1fd05c4ca3b499ddeb7549dbf03988a4e30ea1a8))

## [1.4.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-web-enterprise-v1.3.0-enterprise...chunkr-web-enterprise-v1.4.0-enterprise) (2025-04-29)


### Features

* Added task retries, started at time, border boxes and more UI enhancements for special segment types like tables, formulas and image descriptions, modified self-host tier based usage section and always showing overage ticker ([#460](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/460)) ([40a4a69](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/40a4a6987f82fa01a5cadcbedeee4264bcdb7916))


### Bug Fixes

* Viewer optimizations and momentum scroll fix ([#466](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/466)) ([671f840](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/671f84083eb796b9a120e3ad3f57c7a61cbfcde3))

## [1.3.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-web-enterprise-v1.2.0-enterprise...chunkr-web-enterprise-v1.3.0-enterprise) (2025-04-23)


### Features

* Update web to use `/task/parse` API route and added `error_handling`, `llm_processing`, `embed_sources` and tokenization settings in the upload component ([#450](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/450)) ([13ec8c7](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/13ec8c772ecdb54983fd009be7e59e37b3695ba1))


### Bug Fixes

* Llm models are now fetched from `VITE_API_URL` ([#457](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/457)) ([6f2a261](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/6f2a261fc3389cd257de4818ea302ae9920d837a))

## [1.2.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-web-enterprise-v1.1.0-enterprise...chunkr-web-enterprise-v1.2.0-enterprise) (2025-03-20)


### Features

* Added new cropped image viewing, updated upload component defaults for image VLM processing, and some bug fixes for segment highlighting + JSON viewing ([#388](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/388)) ([6115ee0](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/6115ee08b785e94ed8432e4c75da98e32a42bea9))

## [1.1.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-web-enterprise-v1.0.0-enterprise...chunkr-web-enterprise-v1.1.0-enterprise) (2025-03-13)


### Features

* **core:** Improved image uploads to pdf conversion and added checkbox support ([a2b65ed](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a2b65ed182dcc07af1bccc5b4e98dec3a3335ed8))
