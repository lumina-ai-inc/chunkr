use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use flate2::read::GzDecoder;
use reqwest;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tar::Archive;
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_path")]
    pub path: String,
}

fn default_path() -> String {
    "./pdfium-binaries".to_string()
}

#[derive(Error, Debug)]
pub enum PdfiumError {
    #[error("PDFium binary not found at specified path: {0}")]
    BinaryNotFound(String),
    #[error("Failed to download PDFium binary: {0}")]
    DownloadError(String),
    #[error("Path neither a file nor a directory")]
    PathError(String),
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

    pub async fn ensure_binary(&self) -> Result<(), PdfiumError> {
        let path = Path::new(&self.path);

        if path.is_file() {
            return Ok(());
        }

        let target_path = if path.is_dir() {
            let binary_name = self.os_binary_name()?;
            path.join(binary_name).to_string_lossy().to_string()
        } else {
            return Err(PdfiumError::PathError(
                "Path neither a file nor a directory".to_string(),
            ));
        };

        println!("PDFium binary not found, downloading to {}...", target_path);
        self.download_binary(&target_path).await?;

        Ok(())
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

        if let Some(parent) = Path::new(target_path).parent() {
            fs::create_dir_all(parent).map_err(|e| PdfiumError::DownloadError(e.to_string()))?;
        }

        fs::copy(&source_path, target_path)
            .map_err(|e| PdfiumError::DownloadError(format!("Failed to copy binary: {}", e)))?;

        Ok(())
    }
}
