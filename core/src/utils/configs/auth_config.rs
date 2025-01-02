use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_keycloak_url")]
    pub keycloak_url: String,
    #[serde(default = "default_keycloak_realm")]
    pub keycloak_realm: String,
}

fn default_keycloak_url() -> String {
    "http://localhost:8080".to_string()
}

fn default_keycloak_realm() -> String {
    "chunkr".to_string()
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("AUTH")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
