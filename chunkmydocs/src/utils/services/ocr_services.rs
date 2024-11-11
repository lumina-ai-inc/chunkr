use crate::{
    models::server::segment::OCRResult,
    models::workers::general_ocr::PaddleOCRResponse,
    models::workers::table_ocr::{PaddleTableRecognitionResponse, PaddleTableRecognitionResult},
    utils::configs::concurrent_config::Config as ConcurrentConfig,
    utils::configs::llm_config::{get_prompt, Config as LlmConfig},
    utils::configs::worker_config::Config as WorkerConfig,
    utils::services::llm::vlm_call,
};
use image_base64;
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::path::Path;
use tokio::sync::Semaphore;

static GENERAL_OCR_SEMAPHORE: OnceCell<Semaphore> = OnceCell::new();
static VLM_OCR_SEMAPHORE: OnceCell<Semaphore> = OnceCell::new();

fn init_semaphores() {
    let concurrent_config = ConcurrentConfig::from_env().unwrap();
    GENERAL_OCR_SEMAPHORE.get_or_init(|| Semaphore::new(concurrent_config.general_ocr));
    VLM_OCR_SEMAPHORE.get_or_init(|| Semaphore::new(concurrent_config.vlm_ocr));
}

#[derive(Debug)]
struct OcrError(String);

impl fmt::Display for OcrError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for OcrError {}

pub async fn paddle_ocr(file_path: &Path) -> Result<Vec<OCRResult>, Box<dyn Error + Send + Sync>> {
    init_semaphores();
    let _permit = GENERAL_OCR_SEMAPHORE
        .get()
        .expect("OCR semaphore not initialized")
        .acquire()
        .await?;
    let client = reqwest::Client::new();
    let config = WorkerConfig::from_env()
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
    drop(_permit);
    Ok(ocr_results)
}

pub async fn paddle_table_ocr(
    file_path: &Path,
) -> Result<PaddleTableRecognitionResult, Box<dyn Error + Send + Sync>> {
    let _permit = GENERAL_OCR_SEMAPHORE
        .get()
        .expect("OCR semaphore not initialized")
        .acquire()
        .await?;
    let client = reqwest::Client::new();
    let config = WorkerConfig::from_env()
        .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;

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
    drop(_permit);
    Ok(paddle_table_response.result)
}

pub async fn vlm_table_ocr(file_path: &Path) -> Result<String, Box<dyn Error + Send + Sync>> {
    init_semaphores();
    let _permit = VLM_OCR_SEMAPHORE
        .get()
        .expect("OCR semaphore not initialized")
        .acquire()
        .await?;
    let llm_config = LlmConfig::from_env().unwrap();
    let prompt = get_prompt("table", &HashMap::new())?;
    let response = vlm_call(
        file_path,
        llm_config.ocr_url.unwrap_or(llm_config.url),
        llm_config.ocr_key.unwrap_or(llm_config.key),
        llm_config.ocr_model.unwrap_or(llm_config.model),
        prompt,
        None,
        None,
    )
    .await
    .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;
    drop(_permit);
    Ok(response)
}

pub async fn vlm_formula_ocr(file_path: &Path) -> Result<String, Box<dyn Error + Send + Sync>> {
    init_semaphores();
    let _permit = VLM_OCR_SEMAPHORE
        .get()
        .expect("OCR semaphore not initialized")
        .acquire()
        .await?;
    let llm_config = LlmConfig::from_env().unwrap();
    let prompt = get_prompt("formula", &HashMap::new())?;
    let response = vlm_call(
        file_path,
        llm_config.ocr_url.unwrap_or(llm_config.url),
        llm_config.ocr_key.unwrap_or(llm_config.key),
        llm_config.ocr_model.unwrap_or(llm_config.model),
        prompt,
        None,
        None,
    )
    .await
    .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;
    drop(_permit);
    Ok(response)
}
