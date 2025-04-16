docker run --rm -v "${PWD}:/local" openapitools/openapi-generator-cli generate -i /local/.chunkr/openapi.json -g rust -o /local/clients/rust --skip-validate-spec --package-name chunkr-ai
