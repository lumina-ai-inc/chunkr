# Changelog

## [2.9.7-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.9.6-enterprise...v2.9.7-enterprise) (2025-07-29)


### Bug Fixes

* Added normalization for calamine sheet names when finding corresponding html ([#207](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/207)) ([e7c3885](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e7c38859912223b0088d997466c47581f7bc502f))

## [2.9.6-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.9.5-enterprise...v2.9.6-enterprise) (2025-07-26)


### Bug Fixes

* Sync issues with libreoffice and calamine coordinate system and sheets, add support for empty and visible sheets, filter out empty segments from excel outputs, added simplification of single cell range (e.g. F42:F42 -> F42), and allow no segments to pass through chunking ([#204](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/204)) ([df8a62a](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/df8a62a71597a0c78a859533928b55e231eb8a74))

## [2.9.5-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.9.4-enterprise...v2.9.5-enterprise) (2025-07-25)


### Bug Fixes

* On boarding redirect, register regex & get-tasks ([#201](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/201)) ([8adeaf1](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/8adeaf18cf1284c8a04be9f9dee40b388ef9c2bd))

## [2.9.4-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.9.3-enterprise...v2.9.4-enterprise) (2025-07-24)


### Bug Fixes

* Use cell.text instead of cell.value ([d970d75](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d970d7579c9cda4addc9be627dcbc3979b481d81))

## [2.9.3-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.9.2-enterprise...v2.9.3-enterprise) (2025-07-24)


### Bug Fixes

* Added truncation for excel file elements ([4843a25](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/4843a255e89d6950d92df61c1b2b44b33debe943))

## [2.9.2-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.9.1-enterprise...v2.9.2-enterprise) (2025-07-24)


### Bug Fixes

* Added better deserialization for range ([#192](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/192)) ([69489c0](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/69489c0237ea558fbbad5f48f73b8c0142960587))

## [2.9.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.9.0-enterprise...v2.9.1-enterprise) (2025-07-23)


### Bug Fixes

* Added python client to config ([a248f9b](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a248f9b3be09fbb69689723407e73e4ddf9f7b20))

## [2.9.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.8.1-enterprise...v2.9.0-enterprise) (2025-07-23)


### Features

* Added pypi release ([254d493](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/254d4930e862c4e5c689c9b0e320c28210e77a09))
* Update sdk with new models ([d2fccfd](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d2fccfd493b768c0f52a53f69d4a58a925208412))


### Bug Fixes

* Increase slots, add edu to blocked regex and verify cal endpoints during auth ([#186](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/186)) ([5dd392a](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5dd392a4e7f2496e32b308489fd26efd63f2c414))
* Style error pages for auth ([#184](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/184)) ([3b56cac](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/3b56cac87658cb184cb7e716a5c67ba5c5f0b702))

## [2.8.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.8.0-enterprise...v2.8.1-enterprise) (2025-07-20)


### Bug Fixes

* Split header td not in tr bug ([#180](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/180)) ([446b6be](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/446b6bef5cb4aacde01a882314e42affdab943e6))
* Style update password page ([#182](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/182)) ([70ab411](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/70ab411aa565ea839d75b60da88673a26db58ad7))

## [2.8.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.7.2-enterprise...v2.8.0-enterprise) (2025-07-17)


### Features

* Added styling and formulas to cells and html in the excel parser ([#174](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/174)) ([9231675](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/9231675ab5b979e835a811b844a08a62cad798f1))
* Excel viewer + segment chunk fix ([#176](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/176)) ([84b8792](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/84b8792f248f50799fe7603f9c28bb3382a40471))


### Bug Fixes

* Adjusted segment chunk rendering to reflect new output structure and removed homepage viewer for now ([#172](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/172)) ([6b4ab6b](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/6b4ab6b8e82be19f15fe7bf6b8efe76f0d8cee35))

## [2.7.2-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.7.1-enterprise...v2.7.2-enterprise) (2025-07-15)


### Bug Fixes

* Adjusted upload modal UI layout and fixed segment processing card ([#167](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/167)) ([ac8f9dc](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/ac8f9dc2b05ab8862bdaf67de9ec89a1975244ad))

## [2.7.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.7.0-enterprise...v2.7.1-enterprise) (2025-07-14)


### Bug Fixes

* Adjusted ui to new segment processing API, fixes for picture description viewing, some functionality buttons moved around for cleaner UX, chunks are clearly differentiated ([#163](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/163)) ([a53d525](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a53d525aa632313bbe4add9b3a7e5f38ca35b40e))

## [2.7.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.6.0-enterprise...v2.7.0-enterprise) (2025-07-13)


### Features

* Add rate limiter to Azure requests with configurable limits ([#159](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/159)) ([10991fd](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/10991fd168efe524022073961001a665ae95b96f))

## [2.6.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.5.0-enterprise...v2.6.0-enterprise) (2025-07-11)


### Features

* Add feature to convert pdf to images and improve regex check ([f62abcc](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/f62abcc715e277a720b2c6954bc9f6c1fd83a025))

## [2.5.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.4.0-enterprise...v2.5.0-enterprise) (2025-07-09)


### Features

* Removed getting old artifacts and allows timed out tasks to be retried ([#150](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/150)) ([6b8de90](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/6b8de9051a8b7f0da0822563a669792c6e8f5f2b))


### Bug Fixes

* Added /onboarding to web and updated TEI ([cc223fd](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/cc223fdd8a615098f80d1d165420e18283dbcf45))
* Auth styling changes, add regex in register form & fix redirect logic ([#153](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/153)) ([a09c730](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a09c730c421348f296f655a57e9f05febf0dcefc))
* Image pull secrets being applied properly and updated to auth docker image from keycloak ([1642dc8](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/1642dc88f63285b53a007261dc777ff18abf066e))

## [2.4.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.3.0-enterprise...v2.4.0-enterprise) (2025-07-08)


### Features

* Add a feature flag to include_excel_headers ([#140](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/140)) ([00e225e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/00e225e266d6bd6b385f5f75ffff6b1dbc63f54d))
* Add intake form and restyle keycloak auth pages ([2d8bde8](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/2d8bde81fa201a857d1afec3fe4ba489de741761))
* Added teleport terraform as an ssh alternative([#145](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/145)) ([e464ab9](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e464ab99fb558c0ede131a82ee67817703998871))
* Workload identity added to chunkr pods ([#146](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/146)) ([7f05114](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/7f0511409f3de3984af9460725fcb9de97a3250c))


### Bug Fixes

* Removed api key debug prints for llm, and base64 using external urls ([a4aaa53](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a4aaa535fd79e70e32dff7a2457eef75f7b0004a))
* Updated monthly page usage input for onboarding | 100K+ -&gt; 1M+ ([de7505a](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/de7505a81b95195af8e7a11838ba28ecc1cbb39d))

## [2.3.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.2.0-enterprise...v2.3.0-enterprise) (2025-07-07)


### Features

* Added vertex ai support ([#143](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/143)) ([2ce9327](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/2ce9327a2c02bd4d0a3ca22fc67c24119ceddc28))
* Added workload identity to tf and kube, and add GCP CLI to the docker images  ([#141](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/141)) ([f514cef](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/f514cef95f4d2cfe3e53cbfb0898347a9b4a501b))

## [2.2.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.1.0-enterprise...v2.2.0-enterprise) (2025-07-04)


### Features

* Enhance excel parser to identify all element types with improved error handling ([#133](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/133)) ([9f15fd7](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/9f15fd73b39bd529aba7af245822deeb4dce7cc6))


### Bug Fixes

* Browser errors on kube ([#128](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/128)) ([418af50](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/418af50cffbc34ece97d4de0e51b56ac2bcfdeb5))

## [2.1.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v2.0.0-enterprise...v2.1.0-enterprise) (2025-07-03)


### Features

* Added flag for experimental features, and set default for spreadsheet feature to be off ([455a13e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/455a13ebd05f2a15fa68a2bdefda38d6b367a50e))


### Bug Fixes

* Added chrome sandbox false for docker containers ([181debd](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/181debd637e051610f5ed0e9afd9ae5e8581ba5a))
* Remove keycloakify theme ([ea46cbe](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/ea46cbea3e6d6abca04f2c79f8c5bd75d496ac7b))

## [2.0.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.22.1-enterprise...v2.0.0-enterprise) (2025-06-29)


### ⚠ BREAKING CHANGES

* consolidate HTML/markdown generation into single format choice

### Features

* Add advanced Excel parser with automatic table identification, LibreOffice-based HTML conversion, headless Chrome rendering, and specialized spreadsheet chunking pipeline ([5616c21](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5616c2107f35bdcbd6b6933c878043dc53665611))
* Consolidate HTML/markdown generation into single format choice ([a974f3f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a974f3fbc2bd9158ca052c21a121b479e0eb7613))
* Redesign keycloak auth pages ([adaf74b](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/adaf74b2c57b37779794e9e3c736905b02bccff3))


### Bug Fixes

* Added timeouts to azure and improved error messages ([#119](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/119)) ([f8bb3f1](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/f8bb3f1ae31fd668a34177900a07bd0b8dd09241))
* Changes to landing page cover image and removed code example on landing ([#121](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/121)) ([5bb4a98](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5bb4a98d2bf53b635dbfe8033c20efa6bfad2653))
* Resolve logger warnings ([#510](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/510)) ([8810d5e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/8810d5ec1ce03c12daa1bee98afed3fb2386cf5a))

## [1.22.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.22.0-enterprise...v1.22.1-enterprise) (2025-06-20)


### Bug Fixes

* Updated rust version ([0c5abae](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/0c5abae8eb3e32305e59c942ba0b7de307d700eb))
* Updated rust version ([5fce4c4](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5fce4c4496dc02954088373b415ba7722ff076be))

## [1.22.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.21.1-enterprise...v1.22.0-enterprise) (2025-06-20)


### Features

* Terraform SOC2 compliance ([#114](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/114)) ([d07dc25](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d07dc25ca2ae1194e09974469bc08dbdbadb2461))


### Bug Fixes

* Added security headers for SOC2 ([ff342a9](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/ff342a924321b02257a5ea94b46e4bea5b27c367))
* Pandoc added to dockers and ./docker.sh files updated ([#111](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/111)) ([0ae9f96](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/0ae9f96057fc4a6456f7481beca5a5936bfaa5f6))
* Reverted landing page hero image ([#548](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/548)) ([e798336](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e7983361fdbb9243c055f2444cacb55aa6072a78))
* Stored cross-site scripting in segmentchunk component ([#546](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/546)) ([49334b7](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/49334b788e742f7453c8987e856b57dcb56f0773))

## [1.21.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.21.0-enterprise...v1.21.1-enterprise) (2025-05-29)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#541](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/541)) ([00db663](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/00db66361022d5566f281badf327a73724da7922))
* Update default for high resolution to true ([#536](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/536)) ([37cc757](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/37cc757ea41acce4a662a127bc141e77b56cda03))
* Update robots.txt and blog post page, added sitemap, and removed toast from auth  ([#535](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/535)) ([3c8f827](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/3c8f82701d4ff40f932b24607da2dfd394f31e60))
* Updated manual conversion from html to mkd to pandoc ([#539](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/539)) ([16cc847](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/16cc847f5728bd963bf8f367579721098190141c))


### Performance Improvements

* Improved llm analytics for extraction errors ([#538](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/538)) ([0302499](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/0302499f9942c3f2d0f3888ef419d7e2f6945394))

## [1.21.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.20.4-enterprise...v1.21.0-enterprise) (2025-05-28)


### Features

* Added blog pages to frontend - hooked up to Contentful CMS + env toggle for Blog page ([#531](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/531)) ([90d6dd8](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/90d6dd88aa5e6cd0bb0580185a6f4fbf3523e35d))
* Updated llm processing pipeline to be more robust ([#527](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/527)) ([1cfb9ca](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/1cfb9ca10801df57175e5ca148852a48cfea54ed))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#528](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/528)) ([7a972ed](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/7a972edddbecce0fcc10edc53669784f6755de89))

## [1.20.4-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.20.3-enterprise...v1.20.4-enterprise) (2025-05-24)


### Bug Fixes

* Rm extended context default and picture default ([3772841](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/37728415a99290f57bbc22d8f62bdf025b22adcb))
* Rm extended context default and picture default ([8411eca](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/8411eca5dc9c66833cd431049ada51675d3ce81f))

## [1.20.3-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.20.2-enterprise...v1.20.3-enterprise) (2025-05-23)


### Bug Fixes

* Merge remnant ([ef2dbe5](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/ef2dbe57182fcb85c75a30a021feb5393c17d7f2))
* Picture default strategy to LLM as default in the generation strategy for html and markdown ([#524](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/524)) ([e44126c](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e44126c0387fb176f9ac6b027e3d6d0231102591))

## [1.20.2-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.20.1-enterprise...v1.20.2-enterprise) (2025-05-22)


### Bug Fixes

* Added file type attributes to span ([#520](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/520)) ([edd7d0d](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/edd7d0d140bd3482ce195e6f2243a9e67b4a5efa))

## [1.20.1-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.20.0-enterprise...v1.20.1-enterprise) (2025-05-22)


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#516](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/516)) ([a938056](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a938056779debec5357ad54b27bf5f0788382ba3))
* **core:** Auto-fix clippy warnings ([#518](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/518)) ([238f47f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/238f47fdaf5d2e62d12448424d1018eb1803b8f8))
* Improved error handling and telemetry for segment processing ([#515](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/515)) ([2afc82e](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/2afc82e361387b51a5d5ab5f99cf74978917e9e1))

## [1.20.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.19.0-enterprise...v1.20.0-enterprise) (2025-05-22)


### Features

* **core:** Improved telemetry and added timeout around task processing to deal with long running processes ([#511](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/511)) ([bbe5913](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/bbe59130afffedbf5e2e29267afb1f6300918f67))


### Bug Fixes

* Cleaner refreshes, fuzzy search, and better API key storage ([#96](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/96)) ([121f63d](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/121f63d2d6c4f66cc4aaffdbd96e630efabb5cee))
* Compile errors ([353e79f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/353e79f0b000fb37204a42ba6deedff949ee335d))
* **core:** Auto-fix clippy warnings ([#512](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/512)) ([d9ecf60](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/d9ecf60f308cfe4607673bec172f8fc04d673135))

## [1.19.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.18.0-enterprise...v1.19.0-enterprise) (2025-05-20)


### Features

* Added Open telemetry support for better analytics ([#504](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/504)) ([7baa3d4](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/7baa3d4a03b5bd15c70dd73b00146adf6dfe7ba6))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#507](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/507)) ([a8c2e70](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a8c2e70fd5db6503fb38273b611fb7ea16c00422))
* Repeated requests in analytics web  ([#90](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/90)) ([a4ccca8](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/a4ccca8c953b55466a02049b50a7ad0b0fa7fab3))

## [1.18.0-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.17.3-enterprise...v1.18.0-enterprise) (2025-05-17)


### Features

* Added classification pipeline for PDFs (making data for DLA training) ([#86](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/86)) ([e8ca16c](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e8ca16ca9b16fe02df5bfe8b01e831ce60933fde))
* Added new frontend pdfs in new landing page ([#490](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/490)) ([bbaf911](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/bbaf911f205b2f81b723577155e6b5adff246a65))
* Added span class instructions for all formula prompts ([#489](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/489)) ([5162f8f](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5162f8f02fabe8eb0a0f99de1373c5295d3f9ddd))
* Added task level analytics to usage page ([#498](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/498)) ([e4d63ff](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/e4d63ffb86c9d790c8bb13cf0cf71642d2f19e2b))
* **core:** Improved error messages on task failure and retry only failed steps ([#496](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/496)) ([2e09e11](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/2e09e113f8cf0b0950a77c6954cc9ded2e85c434))


### Bug Fixes

* **core:** Auto-fix clippy warnings ([#497](https://github.com/lumina-ai-inc/chunkr-enterprise/issues/497)) ([8469ddf](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/8469ddf179709c40949965be203259231bf6b950))
* Reverted to 1.17.3 dockers ([1960a89](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/1960a8983dfae846c14e2efb1dd6c023da15923a))

## [1.17.3-enterprise](https://github.com/lumina-ai-inc/chunkr-enterprise/compare/v1.17.2-enterprise...v1.17.3-enterprise) (2025-05-12)


### Bug Fixes

* Address Clippy warnings ([5fee794](https://github.com/lumina-ai-inc/chunkr-enterprise/commit/5fee794ef7bc3cc9fcc6b684a839c67281d7e566))

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
