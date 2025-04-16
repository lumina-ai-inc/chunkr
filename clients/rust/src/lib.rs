use progenitor::generate_api;

generate_api!(
    spec = "../../.chunkr/openapi.json",
    interface = Builder
);
