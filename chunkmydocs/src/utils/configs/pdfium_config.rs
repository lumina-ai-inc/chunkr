use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use flate2::read::GzDecoder;
use reqwest;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tar::Archive;
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_path")]
    pub dir_path: PathBuf,
}

fn default_path() -> PathBuf {
    PathBuf::from("./pdfium-binaries")
}

#[derive(Error, Debug)]
pub enum PdfiumError {
    #[error("PDFium binary not found at specified path: {0}")]
    BinaryNotFound(String),
    #[error("Failed to download PDFium binary: {0}")]
    DownloadError(String),
    #[error("Error creating directory: {0}")]
    DirError(String),
    #[error("Error loading config: {0}")]
    ConfigError(String),
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("PDFIUM")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }

    fn os_binary_name(&self) -> Result<String, PdfiumError> {
        match std::env::consts::OS {
            "windows" => Ok("pdfium.dll.lib".to_string()),
            "linux" => Ok("libpdfium.so".to_string()),
            "macos" => Ok("libpdfium.dylib".to_string()),
            _ => Err(PdfiumError::DownloadError(
                "Unsupported platform".to_string(),
            )),
        }
    }

    pub async fn get_binary(&self) -> Result<String, PdfiumError> {
        let path = self.dir_path.clone();
        let binary_name = self.os_binary_name()?;

        if !path.clone().exists() {
            fs::create_dir_all(path.clone())
                .map_err(|e: std::io::Error| PdfiumError::DirError(e.to_string()))?;
        }

        let target_path = path
            .clone()
            .join(&binary_name)
            .to_string_lossy()
            .to_string();

        if !Path::new(&target_path).exists() {
            println!("PDFium binary not found, downloading to {}...", target_path);
            self.download_binary(&target_path).await?;
        }

        Ok(target_path)
    }

    async fn download_binary(&self, target_path: &str) -> Result<(), PdfiumError> {
        let download_url = match std::env::consts::OS {
            "windows" => "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-windows-x64.tgz",
            "linux" => "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-x64.tgz",
            "macos" => "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-mac-x64.tgz",
            _ => return Err(PdfiumError::DownloadError("Unsupported platform".to_string())),
        };

        let response = reqwest::get(download_url)
            .await
            .map_err(|e| PdfiumError::DownloadError(e.to_string()))?;

        let bytes = response
            .bytes()
            .await
            .map_err(|e| PdfiumError::DownloadError(e.to_string()))?;

        let temp_dir = tempfile::tempdir()
            .map_err(|e| PdfiumError::DownloadError(format!("Failed to create temp dir: {}", e)))?;

        let gz = GzDecoder::new(&bytes[..]);
        let mut archive = Archive::new(gz);
        archive
            .unpack(temp_dir.path())
            .map_err(|e| PdfiumError::DownloadError(format!("Failed to extract archive: {}", e)))?;

        let lib_path = temp_dir.path().join("lib");
        let binary_name = self.os_binary_name()?;

        let source_path = lib_path.join(binary_name);

        fs::copy(&source_path, target_path)
            .map_err(|e| PdfiumError::DownloadError(format!("Failed to copy binary: {}", e)))?;

        Ok(())
    }
}
