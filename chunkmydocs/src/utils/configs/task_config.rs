use config::{ Config as ConfigTrait, ConfigError };
use serde::{ Deserialize, Serialize };
use dotenvy::dotenv_override;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_pdf_density")]
    pub pdf_density: f32,
    #[serde(default = "default_page_image_density")]
    pub page_image_density: f32,
    #[serde(default = "default_segment_bbox_offset")]
    pub segment_bbox_offset: f32,
}

fn default_pdf_density() -> f32 {
    72.0
}

fn default_page_image_density() -> f32 {
    150.0
}

fn default_segment_bbox_offset() -> f32 {
    5.0
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