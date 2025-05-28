# Changelog

## [1.6.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-web-v1.5.1...chunkr-web-v1.6.0) (2025-05-28)


### Features

* Added blog pages to frontend - hooked up to Contentful CMS + env toggle for Blog page ([#531](https://github.com/lumina-ai-inc/chunkr/issues/531)) ([90d6dd8](https://github.com/lumina-ai-inc/chunkr/commit/90d6dd88aa5e6cd0bb0580185a6f4fbf3523e35d))

## [1.5.1](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-web-v1.5.0...chunkr-web-v1.5.1) (2025-05-22)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#518](https://github.com/lumina-ai-inc/chunkr/issues/518)) ([238f47f](https://github.com/lumina-ai-inc/chunkr/commit/238f47fdaf5d2e62d12448424d1018eb1803b8f8))

## [1.5.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-web-v1.4.0...chunkr-web-v1.5.0) (2025-05-17)


### Features

* Added new frontend pdfs in new landing page ([#490](https://github.com/lumina-ai-inc/chunkr/issues/490)) ([bbaf911](https://github.com/lumina-ai-inc/chunkr/commit/bbaf911f205b2f81b723577155e6b5adff246a65))
* Added task level analytics to usage page ([#498](https://github.com/lumina-ai-inc/chunkr/issues/498)) ([e4d63ff](https://github.com/lumina-ai-inc/chunkr/commit/e4d63ffb86c9d790c8bb13cf0cf71642d2f19e2b))

## [1.4.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-web-v1.3.2...chunkr-web-v1.4.0) (2025-05-06)


### Features

* Added extended context (full page + segment) in `segment_processing` ([#480](https://github.com/lumina-ai-inc/chunkr/issues/480)) ([542377b](https://github.com/lumina-ai-inc/chunkr/commit/542377b904aef5fb215bdea3f837315a23eb37de))

## [1.3.2](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-web-v1.3.1...chunkr-web-v1.3.2) (2025-05-01)


### Bug Fixes

* List style points visible now ([#476](https://github.com/lumina-ai-inc/chunkr/issues/476)) ([95a9844](https://github.com/lumina-ai-inc/chunkr/commit/95a98449bf6b6c8f0befd728ceb8206656966b8d))

## [1.3.1](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-web-v1.3.0...chunkr-web-v1.3.1) (2025-04-30)


### Bug Fixes

* Solved latex rendering errors and final viewer optimizations in place ([#472](https://github.com/lumina-ai-inc/chunkr/issues/472)) ([1fd05c4](https://github.com/lumina-ai-inc/chunkr/commit/1fd05c4ca3b499ddeb7549dbf03988a4e30ea1a8))

## [1.3.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-web-v1.2.1...chunkr-web-v1.3.0) (2025-04-29)


### Features

* Added task retries, started at time, border boxes and more UI enhancements for special segment types like tables, formulas and image descriptions, modified self-host tier based usage section and always showing overage ticker ([#460](https://github.com/lumina-ai-inc/chunkr/issues/460)) ([40a4a69](https://github.com/lumina-ai-inc/chunkr/commit/40a4a6987f82fa01a5cadcbedeee4264bcdb7916))


### Bug Fixes

* Viewer optimizations and momentum scroll fix ([#466](https://github.com/lumina-ai-inc/chunkr/issues/466)) ([671f840](https://github.com/lumina-ai-inc/chunkr/commit/671f84083eb796b9a120e3ad3f57c7a61cbfcde3))

## [1.2.1](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-web-v1.2.0...chunkr-web-v1.2.1) (2025-04-23)


### Bug Fixes

* Llm models are now fetched from `VITE_API_URL` ([#457](https://github.com/lumina-ai-inc/chunkr/issues/457)) ([06ce1ea](https://github.com/lumina-ai-inc/chunkr/commit/06ce1eaa98048753fd065ddb00908b54914f4857))

## [1.2.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-web-v1.1.0...chunkr-web-v1.2.0) (2025-04-23)


### Features

* Update web to use `/task/parse` API route and added `error_handling`, `llm_processing`, `embed_sources` and tokenization settings in the upload component ([#450](https://github.com/lumina-ai-inc/chunkr/issues/450)) ([b1a6aef](https://github.com/lumina-ai-inc/chunkr/commit/b1a6aef41ff8d73daa9ba435e37219b98c765524))

## [1.1.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-web-v1.0.0...chunkr-web-v1.1.0) (2025-03-20)


### Features

* Added new cropped image viewing, updated upload component defaults for image VLM processing, and some bug fixes for segment highlighting + JSON viewing ([#388](https://github.com/lumina-ai-inc/chunkr/issues/388)) ([6115ee0](https://github.com/lumina-ai-inc/chunkr/commit/6115ee08b785e94ed8432e4c75da98e32a42bea9))

## 1.0.0 (2025-03-12)


### Features

* **core:** Improved image uploads to pdf conversion and added checkbox support ([a2b65ed](https://github.com/lumina-ai-inc/chunkr/commit/a2b65ed182dcc07af1bccc5b4e98dec3a3335ed8))
