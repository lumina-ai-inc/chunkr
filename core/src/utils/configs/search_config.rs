use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_dense_vector_url")]
    pub dense_vector_url: String,
    #[serde(default = "default_dense_vector_api_key")]
    pub dense_vector_api_key: String,
}

fn default_dense_vector_url() -> String {
    "http://localhost:8007".to_string()
}

fn default_dense_vector_api_key() -> String {
    "".to_string()
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
