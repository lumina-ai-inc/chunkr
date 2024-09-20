use crate::models::server::segment::Segment;
use crate::utils::configs::task_config::Config;
use std::path::Path;
use reqwest::{ multipart, Client as ReqwestClient };
use tokio::sync::OnceCell;

static REQWEST_CLIENT: OnceCell<ReqwestClient> = OnceCell::const_new();

async fn get_reqwest_client() -> &'static ReqwestClient {
    REQWEST_CLIENT.get_or_init(|| async { ReqwestClient::new() }).await
}

pub async fn process_segments(
    pdf_path: &Path,
    segments: &Vec<Segment>
) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
    let client = get_reqwest_client().await;
    let config = Config::from_env()?;
    let url = format!("{}/process", config.task_service_url);

    let file_name = pdf_path
        .file_name()
        .ok_or_else(|| "Invalid file name")?
        .to_str()
        .ok_or_else(|| "Non-UTF8 file name")?;

    let file_content = tokio::fs::read(pdf_path).await?;
    let part = multipart::Part::bytes(file_content).file_name(file_name.to_string());

    let form = multipart::Form
        ::new()
        .part("file", part)
        .text("segments", serde_json::to_string(&segments)?)
        .text("image_density", config.image_density.unwrap_or(300).to_string())
        .text("page_image_extension", config.page_image_extension.unwrap_or("png".to_string()))
        .text("segment_image_extension", config.segment_image_extension.unwrap_or("jpg".to_string()));

    let response = client.post(url).multipart(form).send().await?.error_for_status()?;

    let response_json: Vec<Segment> = response.json().await?;

    Ok(response_json)
}
