use config::{ Config as ConfigTrait, ConfigError };
use serde::{ Deserialize, Serialize };
use dotenvy::dotenv_override;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_self_hosted")]
    pub self_hosted: bool,
}

fn default_self_hosted() -> bool {
    true
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("USER").separator("__"))
            .build()?
            .try_deserialize()
    }
}