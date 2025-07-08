use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub api_key: String,
    pub event_type_id: String,
    #[serde(default = "default_url")]
    pub url: String,
    #[serde(default = "default_booking_api_version")]
    pub booking_api_version: String,
    #[serde(default = "default_slots_api_version")]
    pub slots_api_version: String,
}

fn default_url() -> String {
    "https://api.cal.com/v2".to_string()
}

fn default_booking_api_version() -> String {
    "2024-08-13".to_string()
}

fn default_slots_api_version() -> String {
    "2024-09-04".to_string()
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();
        ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("CAL").separator("__"))
            .build()?
            .try_deserialize()
    }
}
