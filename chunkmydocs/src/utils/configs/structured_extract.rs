use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub embedding_url: String,
    pub embedding_key: String,
    pub llm_url: String,
    pub llm_key: String,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    pub batch_size: i32,
    pub model_name: String,
}

fn default_top_k() -> i32 {
    10
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("STRUCTURED_EXTRACT")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
