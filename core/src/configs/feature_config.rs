use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_excel_parser")]
    pub excel_parser: bool,

    #[serde(default = "default_include_excel_headers")]
    pub include_excel_headers: bool,

    #[serde(default = "default_pdf_conversion")]
    pub pdf_conversion: bool,
}

fn default_excel_parser() -> bool {
    false
}

fn default_include_excel_headers() -> bool {
    true
}

fn default_pdf_conversion() -> bool {
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
