use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_embedding_url")]
    pub embedding_url: String,
    #[serde(default = "default_embedding_key")]
    pub embedding_key: String,
    #[serde(default = "default_llm_url")]
    pub llm_url: String,
    #[serde(default = "default_llm_key")]
    pub llm_key: String,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    #[serde(default = "default_batch_size")]
    pub batch_size: i32,
    #[serde(default = "default_model_name")]
    pub model_name: String,
}

fn default_top_k() -> i32 {
    10
}

fn default_batch_size() -> i32 {
    32
}

fn default_embedding_url() -> String {
    "http://localhost:8007".to_string()
}

fn default_embedding_key() -> String {
    "".to_string()
}

fn default_llm_url() -> String {
    "https://api.openai.com/v1/chat/completions".to_string()
}

fn default_llm_key() -> String {
    "".to_string()
}

fn default_model_name() -> String {
    "gpt-4o".to_string()
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
