# Changelog

## [1.8.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.7.2-enterprise...v1.8.0-enterprise) (2025-03-27)


### Features

* Added force replytos and disabled team inbox ([efd854e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/efd854ed5af4db0ff1f2da47693c69e4428f6023))
* **core:** Remove rrq dependency and improve memory management ([92b70dc](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/92b70dceb1188cec926e415ff295127a3fb085cc))
* New picture prompts ([#405](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/405)) ([d161fa0](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d161fa0820fc03ffaf9bdbbf58c124179548a31a))


### Bug Fixes

* **client:** Polling would error out on httpx.ReadTimeout ([#400](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/400)) ([aea1255](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/aea125533063de8bbddb36741aed5c1c07ba693b))
* Removed rrq from kube and docker compose ([a84284f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a84284f097965239db90d3b24872fffe2c096590))

## [1.7.2-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.7.1-enterprise...v1.7.2-enterprise) (2025-03-21)


### Bug Fixes

* **core:** Allow PDFs based on extension if the pages can be counted ([#396](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/396)) ([cfbfd01](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/cfbfd0155f5fcfb6245acc7dbedb1baa0b12df0b))
* **core:** Auto-fix clippy warnings ([#393](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/393)) ([0605227](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/06052278229f0fe1c6feec44172e9048bf09ecc1))

## [1.7.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.7.0-enterprise...v1.7.1-enterprise) (2025-03-21)


### Bug Fixes

* Fixed prompts and retries for LLMs ([#394](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/394)) ([4b31588](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/4b3158889747214abc00ee35c634659491e1c07d))

## [1.7.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.6.2-enterprise...v1.7.0-enterprise) (2025-03-20)


### Features

* Added new cropped image viewing, updated upload component defaults for image VLM processing, and some bug fixes for segment highlighting + JSON viewing ([#388](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/388)) ([6115ee0](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/6115ee08b785e94ed8432e4c75da98e32a42bea9))


### Bug Fixes

* **core:** Update default generation strategies for Picture and Page segments ([5316485](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5316485aeec2f923f6fb24f9ab1fcab18e275299))

## [1.6.2-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.6.1-enterprise...v1.6.2-enterprise) (2025-03-19)


### Bug Fixes

* Added imagemagick to docker images ([d3ac921](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d3ac9215f0c570269ba16f3855512da606fd3d4c))
* Added imagemagick to docker images ([ced4342](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/ced4342bbfbc734d4c59296804f159efc87d7a26))
* **core:** Auto-fix clippy warnings ([#28](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/28)) ([9fe6d9f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/9fe6d9f2d3dbbe3cf284f39c077f18c0fc902171))
* **core:** Auto-fix clippy warnings ([#386](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/386)) ([ccb56f9](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/ccb56f95212e5840d931893929c6dec648123e34))
* Downgraded cuda version for doctr ([36db353](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/36db353079aaf56fd4613ea13b3c88e7d678e897))
* Downgraded cuda version to 11.8 ([97b3e0b](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/97b3e0bc15bfaa8f0e848f642fdf752c6f13d1b5))
* **helm:** Entreprise chart is now independent ([c360eb4](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/c360eb4eb80c28dacdef83425726c8a3165dc353))
* **helm:** Update web and doctr too ([2e1601b](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/2e1601b7a1cfd6e0b3e2db0f2d6a883256bfbb5e))
* Remove chunkr chart updates from git action ([f3809db](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/f3809db247a8bc77b58ff9a124c126eda6d2e96b))

## [1.6.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.6.0-enterprise...v1.6.1-enterprise) (2025-03-14)


### Bug Fixes

* LLM retry when stop reason is length ([e8b3e7c](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e8b3e7cccb65874afbb00cd24ec0524ead9a1f0c))

## [1.6.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.5.1-enterprise...v1.6.0-enterprise) (2025-03-13)


### Features

* **core:** Added compatibility to Google AI Studio ([#380](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/380)) ([f56b74c](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/f56b74c23d1bb0faf050c54a74437139dc9a6938))
* Merge box and reading order yolo ([3e1b734](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/3e1b734fba92550a4e735869a999230962496c80))
* Reading order on yolo ([0d23470](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/0d234707298292a824ed3a3e1f7d801f422ab101))


### Bug Fixes

* Added yolo layout analysis to enterprise compose ([a2287c2](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a2287c27804c31fb41977edf3464448578b5dc50))
* Correct Rust lint workflow configuration ([0b1a1eb](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/0b1a1ebdf42a2c22ddfcff52fb7356ebb4216287))
* Fixed config to update enterprise helm ([d7f2c60](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d7f2c606f4a33660d59d4a5df269a59f356ddbae))

## [1.5.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.5.0-enterprise...v1.5.1-enterprise) (2025-03-13)


### Bug Fixes

* Updated rust version in dockers ([e20aa51](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e20aa516b395e7312823001efb4061ba4fccd209))

## [1.5.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.4.2-enterprise...v1.5.0-enterprise) (2025-03-13)


### Features

* /health return current version ([627e8c9](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/627e8c9a1160bf4a360f6d0ea0f1376f64344642))
* Added automation of docker build ([#365](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/365)) ([f01cb2f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/f01cb2fc66c104066f1188149cdbbb8390337169))
* Added enterprise helm chart ([3a55975](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/3a55975cc4470e742c2eee3b1e0892180d80a632))
* Added release please for automated releases ([#363](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/363)) ([d808d4e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d808d4e72464b83590dfab73fe973e2f98b4f7e7))
* **core:** Improved image uploads to pdf conversion and added checkbox support ([a2b65ed](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a2b65ed182dcc07af1bccc5b4e98dec3a3335ed8))


### Bug Fixes

* Added all the dockers to be built ([750e0d7](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/750e0d793f5c3a8f9d2d96038f8d97ce727f1b16))
* Added back segmentation docker with self hosted runner ([0984ba2](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/0984ba2710fca19a807985e5a92fbf1e185bbb03))
* Added CHANGELOG-ENTERPRISE.md for release please ([06c647a](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/06c647a890e997ea8542c3bf9a872a2898cebfc1))
* Await was missing in response ([1ad37d8](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/1ad37d851ee0379c13ba663fc8bafb3541e409a2))
* Await was missing in response ([632adce](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/632adce42c7850a788e0e46817e2498724c76890))
* Continue on error on docker build ([aca0b44](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/aca0b4444875a1b053924a60380e6ee44a4dc005))
* Debugging please release ([e574177](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e574177cc28c68e86ab08ac5b83328b393b02bf4))
* Debugging please release with core changes ([558a6f9](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/558a6f9fd86c5d6e53b770dd48909a3a60e7f110))
* Docker builds use root version ([82e1768](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/82e176868e215f550377d9aed91e5b37fd57faba))
* Docker compose files update separately ([15328a2](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/15328a23dfd4399b6a56babb18becd04bf7bdf72))
* Docker compose updated uses pr ([f45abd1](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/f45abd130d4c643c288c3492bb27f6736059dfbf))
* Fix keycloak tag ([df9efa5](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/df9efa5e212a517020e47d66c3820e62ca87acf2))
* Github action now removes v from version before tagging ([6c77a1f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/6c77a1f5f435c362ec62aabb8bd29a78cc7eba1e))
* Image tag updates not full image ([7b8791f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/7b8791f6bdee1e2b5f47496936700de4ddaee537))
* Moved infrastructure from values.yaml to infrastructure.yaml ([e4ba284](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e4ba284b85c3290f585abce36d97c8c9860bdb9a))
* Only trigger docker build after releases created ([676c280](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/676c280e975ea37a8a737876854b0e3aa7006fc2))
* Release-please docker build ([6e1ff43](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/6e1ff43ad0d5780d2f4a6e67b0b2bcc47d8964f6))
* Removed segmenetation from docker build ([5dc9e6e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5dc9e6e5d1687bbe6ab3555f7df5656856a43f34))
* Rmeoved changelog from core ([b658a63](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/b658a6373baee8cba156d7272a8c91accda0e0e8))
* Rmeoved changelog from core ([4f7c9c0](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/4f7c9c0595199ae176c883f230713514b889d9a5))
* Small bugfix ([101c623](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/101c62301994347331382cf33e4e15bfdfae0013))
* Updated changelog paths ([d20b811](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d20b8112fc5043f5eecabf1e72e89412b1b5e7b1))
* Updated rust version for docker builds ([e5a3633](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e5a3633e970dacae3ce08e42f5d7249aed592fa6))
