use config::{ Config as ConfigTrait, ConfigError };
use serde::{ Deserialize, Serialize };
use dotenvy::dotenv_override;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub task_service_url: String,
    pub image_density: Option<u32>,
    pub page_image_extension: Option<String>,
    pub segment_image_extension: Option<String>,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("TASK").separator("__"))
            .build()?
            .try_deserialize()
    }
}