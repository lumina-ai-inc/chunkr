use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_dense_vector_url")]
    pub dense_vector_url: String,
    #[serde(default = "default_dense_vector_api_key")]
    pub dense_vector_api_key: String,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_dense_vector_url() -> String {
    "http://localhost:8003".to_string()
}

fn default_dense_vector_api_key() -> String {
    "".to_string()
}

fn default_batch_size() -> usize {
    32
}

fn default_top_k() -> usize {
    45
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("SEARCH")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
