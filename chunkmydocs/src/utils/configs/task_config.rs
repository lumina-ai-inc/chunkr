use config::{ Config as ConfigTrait, ConfigError };
use serde::{ Deserialize, Serialize };
use dotenvy::dotenv_override;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub service_url: String,
    pub page_image_density: Option<u32>,
    pub page_image_extension: Option<String>,
    pub segment_image_extension: Option<String>,
    pub segment_image_quality: Option<u8>,
    pub segment_image_resize: Option<String>,
    pub segment_bbox_offset: Option<u32>,
    pub num_workers: Option<u32>,
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