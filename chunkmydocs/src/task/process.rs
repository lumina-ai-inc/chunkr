use crate::models::server::segment::Segment;
use reqwest::{multipart, Client as ReqwestClient};

static REQWEST_CLIENT: OnceCell<ReqwestClient> = OnceCell::const_new();

pub async fn convert_pdf_to_png(
    pdf_path: &Path,
    bounding_boxes: Vec<BoundingBox>,
) -> Result<ConversionResponse, Box<dyn std::error::Error>> {
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

    let form = multipart::Form::new()
        .part("file", part)
        .text("bounding_boxes", serde_json::to_string(&bounding_boxes)?);

    let response = client
        .post(url)
        .multipart(form)
        .send()
        .await?
        .error_for_status()?;

    let response_json: ConversionResponse = response.json().await?;

    