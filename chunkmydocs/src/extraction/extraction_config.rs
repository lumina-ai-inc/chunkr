use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv;
use serde::Deserialize;
use std::sync::Once;

static INIT: Once = Once::new();

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub version: String,
    pub extraction_queue: String,
    pub grobid_url: String,
    pub pdla_url: String,
    pub pdla_fast_url: String,
    pub table_ocr_url: String,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        INIT.call_once(|| {
            dotenv().ok();
        });

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("EXTRACTION")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
