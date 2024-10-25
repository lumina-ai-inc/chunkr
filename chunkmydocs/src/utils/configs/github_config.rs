use config::{ Config as ConfigTrait, ConfigError };
use serde::{ Deserialize, Serialize };
use dotenvy::dotenv_override;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub github_token: Option<String>,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("GITHUB").separator("__"))
            .build()?
            .try_deserialize()
    }
}