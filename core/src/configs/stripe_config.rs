use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub api_key: String,
    #[serde(default = "default_invoice_interval")]
    pub invoice_interval: u64,
    pub webhook_secret: String,
    pub starter_price_id: String,
    pub starter_meter_price_id: String,
    pub dev_price_id: String,
    pub dev_meter_price_id: String,
    pub growth_price_id: String,
    pub growth_meter_price_id: String,
    pub return_url: String,
}

fn default_invoice_interval() -> u64 {
    86400
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
