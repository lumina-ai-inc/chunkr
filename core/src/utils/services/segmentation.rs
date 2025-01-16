use crate::configs::worker_config::Config as WorkerConfig;
use crate::models::chunkr::output::{OCRResult, Segment};
use crate::models::chunkr::segmentation::ObjectDetectionResponse;
use crate::utils::clients;
use crate::utils::rate_limit::{SEGMENTATION_RATE_LIMITER, SEGMENTATION_TIMEOUT, TOKEN_TIMEOUT};
use crate::utils::retry::retry_with_backoff;
use reqwest::multipart;
use std::error::Error;
use std::fs;
use tempfile::NamedTempFile;

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
    let mut request = client
        .post(format!("{}/batch_async", worker_config.segmentation_url))
        .multipart(form);

    if let Some(timeout) = SEGMENTATION_TIMEOUT.get() {
        request = request.timeout(std::time::Duration::from_secs(timeout.unwrap()));
    }

    let response = request.send().await?.error_for_status()?;
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
