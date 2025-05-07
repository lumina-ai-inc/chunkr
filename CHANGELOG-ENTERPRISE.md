# Changelog

## [1.17.2-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.17.1-enterprise...v1.17.2-enterprise) (2025-05-07)


### Bug Fixes

* Added libreoffice to new dockers ([0ad0052](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/0ad0052d1bb233a34f5ad683249f30b25e0cff79))

## [1.17.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.17.0-enterprise...v1.17.1-enterprise) (2025-05-06)


### Bug Fixes

* Formula prompt selection bug ([#484](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/484)) ([1762c26](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/1762c264b1c8e8989d51ebfada461fe85c7b6e10))
* Removed vunerabilties from dockers ([cda1141](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/cda1141f2307a9b9f29e456c135b4d405d049a8c))

## [1.17.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.16.2-enterprise...v1.17.0-enterprise) (2025-05-06)


### Features

* Added extended context (full page + segment) in `segment_processing` ([#480](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/480)) ([542377b](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/542377b904aef5fb215bdea3f837315a23eb37de))
* Added MGX terraform and kube ([#66](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/66)) ([d922754](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d9227543f0544722691c7e7235d955a36416d74d))


### Bug Fixes

* Add Default trait implementation to FallbackStrategy enum ([#479](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/479)) ([d9a2eaf](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d9a2eaf86470d82e8dfed9af874d3cc49ca76ba5))

## [1.16.2-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.16.1-enterprise...v1.16.2-enterprise) (2025-05-01)


### Bug Fixes

* New mu trigger ([#69](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/69)) ([7a6993e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/7a6993e1a36860e238a38626b12c7350be909afa))

## [1.16.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.16.0-enterprise...v1.16.1-enterprise) (2025-05-01)


### Bug Fixes

* Solved latex rendering errors and final viewer optimizations in place ([#472](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/472)) ([1fd05c4](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/1fd05c4ca3b499ddeb7549dbf03988a4e30ea1a8))

## [1.16.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.15.0-enterprise...v1.16.0-enterprise) (2025-04-29)


### Features

* Added task retries, started at time, border boxes and more UI enhancements for special segment types like tables, formulas and image descriptions, modified self-host tier based usage section and always showing overage ticker ([#460](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/460)) ([40a4a69](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/40a4a6987f82fa01a5cadcbedeee4264bcdb7916))


### Bug Fixes

* Compose-cpu web server replicas ([#467](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/467)) ([41b2433](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/41b2433e6ce747c21a0e157e3759b7f1a27ec5ad))
* Updated problematic ML dependencies ([#468](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/468)) ([5e60961](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5e6096122d333b832c8fff1437cb47f70979683e))
* Viewer optimizations and momentum scroll fix ([#466](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/466)) ([671f840](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/671f84083eb796b9a120e3ad3f57c7a61cbfcde3))

## [1.15.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.14.0-enterprise...v1.15.0-enterprise) (2025-04-26)


### Features

* Fixed reading of text layer (prevent whitespace in between words) ([#461](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/461)) ([8eba7d3](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/8eba7d36d108c736f0d0ca658cf90716c2c0c544))

## [1.14.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.13.0-enterprise...v1.14.0-enterprise) (2025-04-23)


### Features

* Added vlm training code ([3904cfc](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/3904cfca5cd86c60f4ffd8df9cc6bcd09e470215))
* Local pipe ([dae9caf](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/dae9cafaa15cf007362a88b7a81a6324313a1d87))
* Saved myself from disaster ([eba8ce4](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/eba8ce4c6f46e0a00dc86e4bc5bad664328b8630))
* Update web to use `/task/parse` API route and added `error_handling`, `llm_processing`, `embed_sources`, and tokenization settings in the upload component ([#450](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/450)) ([13ec8c7](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/13ec8c772ecdb54983fd009be7e59e37b3695ba1))


### Bug Fixes

* Improved handling of base64, bytes-like and file-like file content in the python client ([#452](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/452)) ([db87f6c](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/db87f6ca2b725b617b84f788341a9ed613fe12c2))
* Llm models are now fetched from `VITE_API_URL` ([#457](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/457)) ([6f2a261](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/6f2a261fc3389cd257de4818ea302ae9920d837a))

## [1.13.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.12.0-enterprise...v1.13.0-enterprise) (2025-04-18)


### Features

* Enterprise chart now supports  configmap ([38772d0](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/38772d0063dc8575ea6d8cfcd826a932ec51df38))


### Bug Fixes

* Config map in helm chart ([5a08085](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5a08085e3c72647dd0833cdcc5574e99948298d5))
* Paths decodable as base64 string can be used with the python client ([#444](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/444)) ([d544aac](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d544aac952d7a6b45ece09b691ad0d1d4b9454c1))

## [1.12.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.11.0-enterprise...v1.12.0-enterprise) (2025-04-16)


### Features

* Added docqa annotator ([#52](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/52)) ([2b38262](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/2b3826270128c0ef4e51610b18cb883c017485e2))
* Allow users to choose an LLM model to use for `segment_processing` by setting the `llm_processing.model_id` param in the POST and PATCH request. The available models can be configured using the `models.yaml` file. ([#437](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/437)) ([ea526c4](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/ea526c4c48692ae5d8a9ba00b70008ce238a4c14))
* Https support for docker compose using nginx ([#434](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/434)) ([868096c](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/868096c8adbb051243e433b2b1c7f440ac1b5997))

## [1.11.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.10.0-enterprise...v1.11.0-enterprise) (2025-04-07)


### Features

* Added configuration for `error_handling` which allows you to choose between `Fail` or `Continue` on non-critical errors ([0baca0a](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/0baca0a519b44d139f64d02bec754f259ed329de))


### Bug Fixes

* Default trait added to chunk processing ([20c6f15](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/20c6f15bf5ef1a538413147103313e65e1223e47))

## [1.10.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.9.1-enterprise...v1.10.0-enterprise) (2025-03-29)


### Features

* Added dev profile to docker compose ([347a3d0](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/347a3d0e108d2868213c090676abe8db5ea77219))
* Chunking can now be configured using `embed_sources` in `segment_processing.{segment_type}` configuration and allows the choice of pre-configured tokenizers or any huggingface tokenizer by setting the `tokenizer` field in `chunk_processing` ([#420](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/420)) ([d88ac64](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d88ac646ece3935f1c7fcd028bb6c5df0b7d00d3))


### Bug Fixes

* Updated hashmap to lru for caching ([d868c76](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d868c76dd16a6751e3baab43190b81e827e26395))

## [1.9.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.9.0-enterprise...v1.9.1-enterprise) (2025-03-28)


### Bug Fixes

* **services:** YOLO layout analysis 400 error fixed ([#45](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/45)) ([93ea81c](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/93ea81c883cd55cfbcd035e9770021d3840afd9d))
* Updated skew ([67807f5](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/67807f55d54b59050ea013d81b233b515b184bfc))

## [1.9.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.8.0-enterprise...v1.9.0-enterprise) (2025-03-27)


### Features

* Added doctr small dockers ([#407](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/407)) ([9b8a56e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/9b8a56e273f39aa15d3001c6f7ccb707900dd584))


### Bug Fixes

* **core:** Handle null started_at values with COALESCE in timeout job ([d068be8](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d068be82b972a6cd830234448e4bbfe5ebb5245a))
* Fixed timeout query ([97950e5](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/97950e54aaa9c10cc5ce42f75600603c27d73168))

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
* Removed changelog from core ([b658a63](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/b658a6373baee8cba156d7272a8c91accda0e0e8))
* Removed changelog from core ([4f7c9c0](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/4f7c9c0595199ae176c883f230713514b889d9a5))
* Small bugfix ([101c623](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/101c62301994347331382cf33e4e15bfdfae0013))
* Updated changelog paths ([d20b811](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d20b8112fc5043f5eecabf1e72e89412b1b5e7b1))
* Updated rust version for docker builds ([e5a3633](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e5a3633e970dacae3ce08e42f5d7249aed592fa6))
