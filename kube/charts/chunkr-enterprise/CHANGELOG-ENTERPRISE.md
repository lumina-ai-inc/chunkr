# Changelog

## [1.4.2](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-chart-enterprise-v1.4.1...chunkr-chart-enterprise-v1.4.2) (2025-07-24)


### Bug Fixes

* Added better deserialization for range ([#192](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/192)) ([69489c0](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/69489c0237ea558fbbad5f48f73b8c0142960587))

## [1.4.1](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-chart-enterprise-v1.4.0...chunkr-chart-enterprise-v1.4.1) (2025-07-09)


### Bug Fixes

* Added /onboarding to web and updated TEI ([cc223fd](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/cc223fdd8a615098f80d1d165420e18283dbcf45))
* Image pull secrets being applied properly and updated to auth docker image from keycloak ([1642dc8](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/1642dc88f63285b53a007261dc777ff18abf066e))

## [1.4.0](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-chart-enterprise-v1.3.0...chunkr-chart-enterprise-v1.4.0) (2025-07-08)


### Features

* Workload identity added to chunkr pods ([#146](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/146)) ([7f05114](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/7f0511409f3de3984af9460725fcb9de97a3250c))

## [1.3.0](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-chart-enterprise-v1.2.4...chunkr-chart-enterprise-v1.3.0) (2025-07-07)


### Features

* Added workload identity to tf and kube, and add GCP CLI to the docker images  ([#141](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/141)) ([f514cef](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/f514cef95f4d2cfe3e53cbfb0898347a9b4a501b))

## [1.2.4](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-chart-enterprise-v1.2.3...chunkr-chart-enterprise-v1.2.4) (2025-07-04)


### Bug Fixes

* Browser errors on kube ([#128](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/128)) ([418af50](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/418af50cffbc34ece97d4de0e51b56ac2bcfdeb5))

## [1.2.3](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-chart-enterprise-v1.2.2...chunkr-chart-enterprise-v1.2.3) (2025-07-03)


### Bug Fixes

* Added chrome sandbox false for docker containers ([181debd](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/181debd637e051610f5ed0e9afd9ae5e8581ba5a))

## [1.2.2](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-chart-enterprise-v1.2.1...chunkr-chart-enterprise-v1.2.2) (2025-06-29)


### Bug Fixes

* Changes to landing page cover image and removed code example on landing ([#121](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/121)) ([5bb4a98](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5bb4a98d2bf53b635dbfe8033c20efa6bfad2653))

## [1.2.1](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-chart-enterprise-v1.2.0...chunkr-chart-enterprise-v1.2.1) (2025-05-22)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#518](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/518)) ([238f47f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/238f47fdaf5d2e62d12448424d1018eb1803b8f8))

## [1.2.0](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-chart-enterprise-v1.1.0...chunkr-chart-enterprise-v1.2.0) (2025-05-06)


### Features

* Added MGX terraform and kube ([#66](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/66)) ([d922754](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d9227543f0544722691c7e7235d955a36416d74d))

## [1.1.0](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-chart-enterprise-v1.0.2...chunkr-chart-enterprise-v1.1.0) (2025-04-18)


### Features

* Enterprise chart now supports  configmap ([38772d0](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/38772d0063dc8575ea6d8cfcd826a932ec51df38))

## [1.0.2](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-chart-enterprise-v1.0.1...chunkr-chart-enterprise-v1.0.2) (2025-03-28)


### Bug Fixes

* **services:** YOLO layout analysis 400 error fixed ([#45](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/45)) ([93ea81c](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/93ea81c883cd55cfbcd035e9770021d3840afd9d))
* Updated skew ([67807f5](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/67807f55d54b59050ea013d81b233b515b184bfc))

## [1.0.1](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/chunkr-chart-enterprise-v1.0.0...chunkr-chart-enterprise-v1.0.1) (2025-03-27)


### Bug Fixes

* Removed rrq from kube and docker compose ([a84284f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a84284f097965239db90d3b24872fffe2c096590))

## 1.0.0 (2025-03-19)


### Features

* Added enterprise helm chart ([3a55975](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/3a55975cc4470e742c2eee3b1e0892180d80a632))
* **core:** Improved image uploads to pdf conversion and added checkbox support ([a2b65ed](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a2b65ed182dcc07af1bccc5b4e98dec3a3335ed8))


### Bug Fixes

* **helm:** Entreprise chart is now independent ([c360eb4](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/c360eb4eb80c28dacdef83425726c8a3165dc353))
