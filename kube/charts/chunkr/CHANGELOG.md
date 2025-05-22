# Changelog

## [1.3.2](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-chart-v1.3.1...chunkr-chart-v1.3.2) (2025-05-22)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#518](https://github.com/lumina-ai-inc/chunkr/issues/518)) ([238f47f](https://github.com/lumina-ai-inc/chunkr/commit/238f47fdaf5d2e62d12448424d1018eb1803b8f8))

## [1.3.1](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-chart-v1.3.0...chunkr-chart-v1.3.1) (2025-04-18)


### Bug Fixes

* Config map in helm chart ([5a08085](https://github.com/lumina-ai-inc/chunkr/commit/5a08085e3c72647dd0833cdcc5574e99948298d5))

## [1.3.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-chart-v1.2.0...chunkr-chart-v1.3.0) (2025-04-15)


### Features

* Allow users to choose an LLM model to use for `segment_processing` by setting the `llm_processing.model_id` param in the POST and PATCH request. The available models can be configured using the `models.yaml` file. ([#437](https://github.com/lumina-ai-inc/chunkr/issues/437)) ([ea526c4](https://github.com/lumina-ai-inc/chunkr/commit/ea526c4c48692ae5d8a9ba00b70008ce238a4c14))

## [1.2.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-chart-v1.1.0...chunkr-chart-v1.2.0) (2025-03-27)


### Features

* Added doctr small dockers ([#407](https://github.com/lumina-ai-inc/chunkr/issues/407)) ([9b8a56e](https://github.com/lumina-ai-inc/chunkr/commit/9b8a56e273f39aa15d3001c6f7ccb707900dd584))

## [1.1.0](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-chart-v1.0.2...chunkr-chart-v1.1.0) (2025-03-27)


### Features

* **core:** Remove rrq dependency and improve memory management ([92b70dc](https://github.com/lumina-ai-inc/chunkr/commit/92b70dceb1188cec926e415ff295127a3fb085cc))

## [1.0.2](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-chart-v1.0.1...chunkr-chart-v1.0.2) (2025-03-13)


### Bug Fixes

* Fix keycloak tag ([df9efa5](https://github.com/lumina-ai-inc/chunkr/commit/df9efa5e212a517020e47d66c3820e62ca87acf2))

## [1.0.1](https://github.com/lumina-ai-inc/chunkr/compare/chunkr-chart-v1.0.0...chunkr-chart-v1.0.1) (2025-03-12)


### Bug Fixes

* Moved infrastructure from values.yaml to infrastructure.yaml ([e4ba284](https://github.com/lumina-ai-inc/chunkr/commit/e4ba284b85c3290f585abce36d97c8c9860bdb9a))

## 1.0.0 (2025-03-12)


### Features

* **core:** Improved image uploads to pdf conversion and added checkbox support ([a2b65ed](https://github.com/lumina-ai-inc/chunkr/commit/a2b65ed182dcc07af1bccc5b4e98dec3a3335ed8))
