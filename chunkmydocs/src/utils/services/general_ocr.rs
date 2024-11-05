use crate::{
    models::server::segment::OCRResult, models::workers::general_ocr::PaddleOCRResponse,
    utils::configs::extraction_config::Config,
};
use image_base64;
use std::error::Error;
use std::fmt;
use std::path::Path;

#[derive(Debug)]
struct OcrError(String);

impl fmt::Display for OcrError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for OcrError {}

pub async fn paddle_ocr(file_path: &Path) -> Result<Vec<OCRResult>, Box<dyn Error + Send + Sync>> {
    let client = reqwest::Client::new();
    let config = Config::from_env()
        .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;

    let paddle_ocr_url = config
        .general_ocr_url
        .ok_or_else(|| "General OCR URL is not set in config".to_string())?;

    let url = format!("{}/ocr", &paddle_ocr_url);

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

    let paddle_ocr_result: PaddleOCRResponse = response.json().await?;
    let ocr_results: Vec<OCRResult> = paddle_ocr_result
        .result
        .texts
        .into_iter()
        .map(|text| OCRResult::from(text))
        .collect();
    Ok(ocr_results)
}
