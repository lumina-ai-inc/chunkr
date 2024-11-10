use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_url")]
    pub url: String,
    #[serde(default = "default_api_key")]
    pub api_key: String,
    pub ocr_model: Option<String>,
    pub ocr_url: Option<String>,
    pub ocr_api_key: Option<String>,
    pub structured_extraction_model: Option<String>,
    pub structured_extraction_url: Option<String>,
    pub structured_extraction_api_key: Option<String>,
}

fn default_model() -> String {
    "gpt-4o".to_string()
}

fn default_url() -> String {
    "https://api.openai.com/v1/chat/completions".to_string()
}

fn default_api_key() -> String {
    "".to_string()
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();
        let config = ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("LLM").separator("__"))
            .build()?
            .try_deserialize::<Self>()?;

        Ok(Self {
            ocr_model: config
                .ocr_model
                .filter(|s| !s.is_empty())
                .or(Some(config.model.clone())),
            ocr_url: config
                .ocr_url
                .filter(|s| !s.is_empty())
                .or(Some(config.url.clone())),
            ocr_api_key: config
                .ocr_api_key
                .filter(|s| !s.is_empty())
                .or(Some(config.api_key.clone())),
            structured_extraction_model: config
                .structured_extraction_model
                .filter(|s| !s.is_empty())
                .or(Some(config.model.clone())),
            structured_extraction_url: config
                .structured_extraction_url
                .filter(|s| !s.is_empty())
                .or(Some(config.url.clone())),
            structured_extraction_api_key: config
                .structured_extraction_api_key
                .filter(|s| !s.is_empty())
                .or(Some(config.api_key.clone())),
            ..config
        })
    }
}
