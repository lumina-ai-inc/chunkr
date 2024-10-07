use crate::models::server::segment::Segment;
use crate::models::server::extract::ExtractionPayload;
use crate::utils::configs::task_config::Config;
use std::path::Path;
use reqwest::{ multipart, Client as ReqwestClient };
use tokio::sync::OnceCell;

static REQWEST_CLIENT: OnceCell<ReqwestClient> = OnceCell::const_new();

async fn get_reqwest_client() -> &'static ReqwestClient {
    REQWEST_CLIENT.get_or_init(|| async { ReqwestClient::new() }).await
}

pub async fn process_segments(
    file_path: &Path,
    segments: &Vec<Segment>,
    extraction_payload: &ExtractionPayload
) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
    let client = get_reqwest_client().await;
    let config = Config::from_env()?;
    let url = format!("{}/process", config.service_url);

    let file_name = file_path
        .file_name()
        .ok_or_else(|| "Invalid file name")?
        .to_str()
        .ok_or_else(|| "Non-UTF8 file name")?;

    let file_content = tokio::fs::read(file_path).await?;
    let part = multipart::Part::bytes(file_content).file_name(file_name.to_string());

    let user_id = extraction_payload.user_id.clone();
    let task_id = extraction_payload.task_id.clone();
    let ocr_strategy = extraction_payload.configuration.ocr_strategy.clone();
    let image_folder_location = extraction_payload.image_folder_location.clone();

    let mut form = multipart::Form
        ::new()
        .part("file", part)
        .text("segments", serde_json::to_string(&segments)?)
        .text("user_id", user_id.to_string())
        .text("task_id", task_id.to_string())
        .text("ocr_strategy", ocr_strategy.to_string())
        .text("image_folder_location", image_folder_location.to_string());

    if let Some(density) = config.page_image_density {
        form = form.text("page_image_density", density.to_string());
    }
    if let Some(extension) = config.page_image_extension {
        form = form.text("page_image_extension", extension);
    }
    if let Some(extension) = config.segment_image_extension {
        form = form.text("segment_image_extension", extension);
    }
    if let Some(quality) = config.segment_image_quality {
        form = form.text("segment_image_quality", quality.to_string());
    }
    if let Some(resize) = &config.segment_image_resize {
        form = form.text("segment_image_resize", resize.clone());
    }
    if let Some(offset) = config.segment_bbox_offset {
        form = form.text("segment_bbox_offset", offset.to_string());
    }
    if let Some(workers) = config.num_workers {
        form = form.text("num_workers", workers.to_string());
    }

    let response = client.post(url).multipart(form).send().await?.error_for_status()?;
    let response_json: Vec<Segment> = response.json().await?;
    Ok(response_json)
}
