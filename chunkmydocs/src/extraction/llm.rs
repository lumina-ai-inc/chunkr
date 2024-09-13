use crate::extraction::pdf2png::ConversionResponse;
use crate::models::server::segment::PngPage;

use crate::models::server::llm::LLMConfig;
use crate::utils::configs::extraction_config::Config;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;

pub async fn apply_llm(
    config: LLMConfig,
    extraction_config: &Config,
    png_pages: Vec<PngPage>,
    prompt: String,
) -> Result<String, Box<dyn Error>> {
    let client = Client::new();
    let url = config
        .model
        .base_url(extraction_config)
        .ok_or_else(|| Box::<dyn Error>::from("Invalid URL"))?;

    let mut form = reqwest::multipart::Form::new()
        .text("prompt", format!("{}", prompt))
        .text("temperature", config.temperature.to_string())
        .text("max_tokens", config.max_tokens.to_string());

    for png_page in png_pages {
        let image_data = base64::decode(&png_page.base64_png)?;
        let part = reqwest::multipart::Part::bytes(image_data)
            .file_name(format!("{}.png", png_page.bb_id))
            .mime_str("image/png")?;
        form = form.part("images", part);
    }

    let response = client.post(url).multipart(form).send().await?;
    let response_body = response.text().await?;
    Ok(response_body)
}
