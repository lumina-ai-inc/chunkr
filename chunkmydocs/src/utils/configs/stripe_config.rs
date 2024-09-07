use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub api_key: String,
    pub page_fast_price_id: String,
    pub page_high_quality_price_id: String,
    pub segment_price_id: String,
    pub webhook_secret: String,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("STRIPE")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
