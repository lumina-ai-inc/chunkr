use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_enable_excel_parse")]
    pub enable_excel_parse: bool,
}

fn default_enable_excel_parse() -> bool {
    false
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("FEATURE")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
