use config::{ Config as ConfigTrait, ConfigError };
use serde::{ Deserialize, Serialize };
use dotenvy::dotenv_override;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub path: String
}

#[derive(Error, Debug)]
pub enum PdfiumError {
    #[error("PDFium binary not found at specified path: {0}")]
    BinaryNotFound(String),
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("PDFIUM").separator("__"))
            .build()?
            .try_deserialize()
    }
    
    pub fn validate(&self) -> Result<(), PdfiumError> {
        if !Path::new(&self.path).exists() {
            return Err(PdfiumError::BinaryNotFound(self.path.clone()));
        }
        Ok(())
    }
}