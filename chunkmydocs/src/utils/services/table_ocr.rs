use crate::{
    models::workers::table_ocr::{PaddleTableRecognitionResponse, PaddleTableRecognitionResult},
    utils::configs::extraction_config::Config,
};
use image_base64;
use std::error::Error;
use std::fmt;
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
) -> Result<PaddleTableRecognitionResult, Box<dyn Error + Send + Sync>> {
    let client = reqwest::Client::new();
    let config = Config::from_env()
        .map_err(|e| Box::new(TableOcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;

    let paddle_table_ocr_url = config
        .table_ocr_url
        .ok_or_else(|| "Paddle table OCR URL is not set in config".to_string())?;

    let url = format!("{}/table-recognition", &paddle_table_ocr_url);

    let mut b64 = image_base64::to_base64(file_path.to_str().unwrap());
    if let Some(index) = b64.find(";base64,") {
        b64 = b64[index + 8..].to_string();
    }
    let payload = serde_json::json!({ "image": b64 });

    let response = client
        .post(&url)
        .json(&payload)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await?
        .error_for_status()?;

    let paddle_table_response: PaddleTableRecognitionResponse = match response.json().await {
        Ok(response) => response,
        Err(e) => {
            return Err(format!("Error parsing table OCR response: {}", e).into());
        }
    };

    Ok(paddle_table_response.result)
}
