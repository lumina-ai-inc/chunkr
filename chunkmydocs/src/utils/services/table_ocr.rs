use crate::{
    models::workers::table_ocr::PaddleTableRecognitionResponse,
    utils::configs::extraction_config::Config,
};
use base64::prelude::*;
use std::error::Error;
use std::fmt;
use std::io::Read;
use std::path::Path;

#[derive(Debug)]
struct TableOcrError(String);

impl fmt::Display for TableOcrError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for TableOcrError {}

pub async fn paddle_table_ocr(
    file_path: &Path,
) -> Result<Vec<PaddleTableRecognitionResponse>, Box<dyn Error + Send + Sync>> {
    let client = reqwest::Client::new();
    let config = Config::from_env()
        .map_err(|e| Box::new(TableOcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;
    
    let paddle_table_ocr_url = config
        .paddle_table_ocr_url
        .ok_or_else(|| format!("Paddle table OCR URL is not set in config"))?;

    let url = format!("{:?}/table-recognition", &paddle_table_ocr_url);

    let mut file = std::fs::File::open(file_path)?;
    let mut file_fs = Vec::new();
    file.read_to_end(&mut file_fs)?;
    let b64 = BASE64_STANDARD.encode(&file_fs);

    let payload = serde_json::json!({ "image": b64 });

    let response = client
        .post(&url)
        .json(&payload)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await?
        .error_for_status()?;

    let paddle_table_response: Vec<PaddleTableRecognitionResponse> = response.json().await?;
    Ok(paddle_table_response)
}
