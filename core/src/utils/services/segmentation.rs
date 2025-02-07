use crate::configs::worker_config::Config as WorkerConfig;
use crate::models::chunkr::output::{OCRResult, Segment};
use crate::models::chunkr::segmentation::ObjectDetectionResponse;
use crate::utils::clients;
use crate::utils::rate_limit::{SEGMENTATION_RATE_LIMITER, SEGMENTATION_TIMEOUT};
use crate::utils::retry::retry_with_backoff;
use reqwest::multipart;
use std::error::Error;
use std::fs;
use tempfile::NamedTempFile;

async fn vgt_segmentation_batch(
    temp_files: &Vec<&NamedTempFile>,
    ocr_results: Vec<Vec<OCRResult>>,
    page_offset: usize,
) -> Result<Vec<Vec<Segment>>, Box<dyn Error + Send + Sync>> {
    let worker_config = WorkerConfig::from_env()?;
    let client = clients::get_reqwest_client();

    let mut form = multipart::Form::new();

    for temp_file in temp_files {
        let file_fs = fs::read(temp_file.path()).expect("Failed to read file");
        form = form.part(
            "files",
            multipart::Part::bytes(file_fs)
                .file_name("image.jpg")
                .mime_str("image/jpeg")?,
        );
    }

    form = form.text(
        "ocr_data",
        serde_json::to_string(&serde_json::json!({
            "data": ocr_results
        }))?,
    );

    let mut request = client
        .post(format!("{}/batch", worker_config.segmentation_url))
        .multipart(form);

    if let Some(timeout) = SEGMENTATION_TIMEOUT.get() {
        if let Some(timeout_value) = timeout {
            request = request.timeout(std::time::Duration::from_secs(*timeout_value));
        }
    }

    let response = request.send().await?.error_for_status()?;
    let object_detection_responses: Vec<ObjectDetectionResponse> = response.json().await?;

    let segments_batch: Vec<Vec<Segment>> = object_detection_responses
        .into_iter()
        .enumerate()
        .map(|(page_idx, resp)| {
            resp.instances.to_segments(
                (page_idx + 1 + page_offset) as u32,
                ocr_results[page_idx].clone(),
            )
        })
        .collect();

    Ok(segments_batch)
}

pub async fn perform_segmentation_batch(
    temp_files: Vec<&NamedTempFile>,
    ocr_results: Vec<Vec<OCRResult>>,
    page_offset: usize,
) -> Result<Vec<Vec<Segment>>, Box<dyn Error + Send + Sync>> {
    Ok(retry_with_backoff(|| async {
        SEGMENTATION_RATE_LIMITER
            .get()
            .unwrap()
            .acquire_token()
            .await?;
        vgt_segmentation_batch(&temp_files, ocr_results.clone(), page_offset).await
    })
    .await?)
}
