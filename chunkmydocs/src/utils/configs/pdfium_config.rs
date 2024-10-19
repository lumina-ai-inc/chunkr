use config::{ Config as ConfigTrait, ConfigError };
use serde::{ Deserialize, Serialize };
use dotenvy::dotenv_override;

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

        let config = ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("PDFIUM").separator("__"))
            .build()?
            .try_deserialize();

        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), PdfiumError> {
        if !Path::new(&self.path).exists() {
            return Err(PdfiumError::BinaryNotFound(self.path.clone()));
        }
        Ok(())
    }
}