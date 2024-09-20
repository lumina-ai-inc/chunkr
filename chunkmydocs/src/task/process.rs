use crate::models::server::segment::Segment;
use crate::utils::configs::task_config::Config;
use reqwest::{ multipart, Client as ReqwestClient };
use tokio::sync::OnceCell;

static REQWEST_CLIENT: OnceCell<ReqwestClient> = OnceCell::const_new();

async fn get_reqwest_client() -> &'static ReqwestClient {
    REQWEST_CLIENT.get_or_init(|| async { ReqwestClient::new() }).await
}

pub async fn process_segments(
    segments: &Vec<Segment>
) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
    let client = get_reqwest_client().await;
    let config = Config::from_env()?;
    let url = format!("{}/convert", config.url);

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
        .text("bounding_boxes", serde_json::to_string(&bounding_boxes)?);

    let response = client.post(url).multipart(form).send().await?.error_for_status()?;

    let response_json: ConversionResponse = response.json().await?;

    Ok(response_json)
}
