use crate::configs::redis_config::{create_pool as create_redis_pool, Pool};
use crate::configs::worker_config::Config as WorkerConfig;
use crate::models::chunkr::output::{OCRResult, Segment};
use crate::models::chunkr::segmentation::ObjectDetectionResponse;
use crate::utils::clients;
use crate::utils::rate_limit::{create_segmentation_rate_limiter, RateLimiter};
use crate::utils::retry::retry_with_backoff;
use once_cell::sync::OnceCell;
use reqwest::multipart;
use std::error::Error;
use std::fs;
use tempfile::NamedTempFile;

static SEGMENTATION_RATE_LIMITER: OnceCell<RateLimiter> = OnceCell::new();
static POOL: OnceCell<Pool> = OnceCell::new();
static SEGMENTATION_TIMEOUT: OnceCell<u64> = OnceCell::new();
static TOKEN_TIMEOUT: OnceCell<u64> = OnceCell::new();

fn init_throttle() {
    POOL.get_or_init(|| create_redis_pool());
    SEGMENTATION_RATE_LIMITER.get_or_init(|| {
        create_segmentation_rate_limiter(POOL.get().unwrap().clone(), "segmentation")
    });
    SEGMENTATION_TIMEOUT.get_or_init(|| 120);
    TOKEN_TIMEOUT.get_or_init(|| 10000);
}

async fn vgt_segmentation(
    temp_file: &NamedTempFile,
    ocr_results: Vec<OCRResult>,
    page_number: u32,
) -> Result<Vec<Segment>, Box<dyn Error + Send + Sync>> {
    let worker_config = WorkerConfig::from_env()?;
    let client = clients::get_reqwest_client();
    let file_fs = fs::read(temp_file.path()).expect("Failed to read file");
    let file_name = temp_file.path().file_name().unwrap().to_str().unwrap();
    let part = multipart::Part::bytes(file_fs).file_name(file_name.to_string());
    let form = multipart::Form::new().part("file", part).text(
        "ocr_data",
        serde_json::to_string(&serde_json::json!({
            "data": ocr_results
        }))?,
    );
    let response = client
        .post(format!("{}/batch_async", worker_config.segmentation_url))
        .multipart(form)
        .timeout(std::time::Duration::from_secs(
            *SEGMENTATION_TIMEOUT.get().unwrap(),
        ))
        .send()
        .await?
        .error_for_status()?;
    let object_detection_response: ObjectDetectionResponse = response.json().await?;
    let segments = object_detection_response
        .instances
        .to_segments(page_number, ocr_results);
    Ok(segments)
}

pub async fn perform_segmentation(
    temp_file: &NamedTempFile,
    ocr_results: Vec<OCRResult>,
    page_number: u32,
) -> Result<Vec<Segment>, Box<dyn Error + Send + Sync>> {
    init_throttle();
    let rate_limiter = SEGMENTATION_RATE_LIMITER.get().unwrap();
    Ok(retry_with_backoff(|| async {
        rate_limiter
            .acquire_token_with_timeout(std::time::Duration::from_secs(
                *TOKEN_TIMEOUT.get().unwrap(),
            ))
            .await?;
        vgt_segmentation(temp_file, ocr_results.clone(), page_number).await
    })
    .await?)
}
