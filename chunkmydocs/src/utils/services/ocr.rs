use crate::{
    models::server::segment::OCRResult,
    models::workers::general_ocr::PaddleOCRResponse,
    models::workers::open_ai::MessageContent,
    models::workers::table_ocr::{PaddleTableRecognitionResponse, PaddleTableRecognitionResult},
    utils::configs::llm_config::{get_prompt, Config as LlmConfig},
    utils::configs::worker_config::Config as WorkerConfig,
    utils::db::deadpool_redis::{create_pool as create_redis_pool, Pool},
    utils::rate_limit::{create_general_ocr_rate_limiter, create_llm_rate_limiter, RateLimiter},
    utils::services::llm::{get_basic_image_message, open_ai_call},
};
use image_base64;
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::path::Path;

static GENERAL_OCR_RATE_LIMITER: OnceCell<RateLimiter> = OnceCell::new();
static LLM_OCR_RATE_LIMITER: OnceCell<RateLimiter> = OnceCell::new();
static POOL: OnceCell<Pool> = OnceCell::new();

fn init_throttle() {
    let llm_config = LlmConfig::from_env().unwrap();
    let llm_ocr_url = llm_config.ocr_url.unwrap_or(llm_config.url);
    let domain_name = llm_ocr_url
        .split("://")
        .nth(1)
        .unwrap_or("llm-ocr")
        .split('/')
        .next()
        .unwrap_or("llm-ocr");
    POOL.get_or_init(|| create_redis_pool());
    GENERAL_OCR_RATE_LIMITER.get_or_init(|| {
        create_general_ocr_rate_limiter(POOL.get().unwrap().clone(), "general_ocr")
    });
    LLM_OCR_RATE_LIMITER
        .get_or_init(|| create_llm_rate_limiter(POOL.get().unwrap().clone(), domain_name));
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
    init_throttle();
    let rate_limiter = GENERAL_OCR_RATE_LIMITER.get().unwrap();
    rate_limiter
        .acquire_token_with_timeout(std::time::Duration::from_secs(30))
        .await?;
    let client = reqwest::Client::new();
    let worker_config = WorkerConfig::from_env()
        .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;

    let paddle_ocr_url = worker_config
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
        .timeout(std::time::Duration::from_secs(300))
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

pub async fn paddle_table_ocr(
    file_path: &Path,
) -> Result<PaddleTableRecognitionResult, Box<dyn Error + Send + Sync>> {
    init_throttle();
    let rate_limiter = GENERAL_OCR_RATE_LIMITER.get().unwrap();
    rate_limiter
        .acquire_token_with_timeout(std::time::Duration::from_secs(30))
        .await?;
    let client = reqwest::Client::new();
    let worker_config = WorkerConfig::from_env()
        .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;

    let paddle_table_ocr_url = worker_config
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

async fn llm_ocr(file_path: &Path, prompt: String) -> Result<String, Box<dyn Error + Send + Sync>> {
    init_throttle();
    let rate_limiter = LLM_OCR_RATE_LIMITER.get().unwrap();
    rate_limiter
        .acquire_token_with_timeout(std::time::Duration::from_secs(30))
        .await?;
    let llm_config = LlmConfig::from_env().unwrap();
    let messages = get_basic_image_message(file_path, prompt)
        .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;
    let response = open_ai_call(
        llm_config.ocr_url.unwrap_or(llm_config.url),
        llm_config.ocr_key.unwrap_or(llm_config.key),
        llm_config.ocr_model.unwrap_or(llm_config.model),
        messages,
        None,
        None,
    )
    .await
    .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;
    if let MessageContent::String { content } = response.choices[0].message.content.clone() {
        Ok(content)
    } else {
        Err("Invalid content type".into())
    }
}

pub async fn llm_table_ocr(file_path: &Path) -> Result<String, Box<dyn Error + Send + Sync>> {
    let prompt = get_prompt("table", &HashMap::new())?;
    llm_ocr(file_path, prompt).await
}

pub async fn llm_formula_ocr(file_path: &Path) -> Result<String, Box<dyn Error + Send + Sync>> {
    let prompt = get_prompt("formula", &HashMap::new())?;
    llm_ocr(file_path, prompt).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::configs::throttle_config::Config as ThrottleConfig;

    #[tokio::test]
    async fn test_general_ocr() -> Result<(), Box<dyn Error + Send + Sync>> {
        let input_dir = Path::new("input");
        let first_image = std::fs::read_dir(input_dir)?
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    let path = e.path();
                    if let Some(ext) = path.extension() {
                        if ext == "jpg" || ext == "jpeg" || ext == "png" {
                            Some(path)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
            })
            .next()
            .ok_or("No image files found in input directory")?;

        let mut tasks = Vec::new();
        let count = 100;
        for _ in 0..count {
            let input_file = first_image.clone();
            let task = tokio::spawn(async move {
                match paddle_ocr(&input_file).await {
                    Ok(_) => Ok(()),
                    Err(e) => {
                        println!("Error processing {:?}: {:?}", input_file, e);
                        Err(e)
                    }
                }
            });
            tasks.push(task);
        }

        let start = std::time::Instant::now();
        let mut error_count = 0;
        for task in tasks {
            if let Err(e) = task.await? {
                println!("Error processing: {:?}", e);
                error_count += 1;
            }
        }
        let duration = start.elapsed();
        let images_per_second = count as f64 / duration.as_secs_f64();
        let throttle_config = ThrottleConfig::from_env().unwrap();
        println!(
            "General OCR rate limit: {:?}",
            throttle_config.general_ocr_rate_limit
        );
        println!("Time taken: {:?}", duration);
        println!("Images per second: {:?}", images_per_second);
        println!("Error count: {:?}", error_count);

        if error_count > 0 {
            Err(format!("Error count {} > 0", error_count).into())
        } else {
            Ok(())
        }
    }
}
