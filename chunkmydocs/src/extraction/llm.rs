use reqwest::Client;
use std::error::Error;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use crate::models::server::{segment::{PngPage, SegmentType, Segment}, llm::LLMConfig};
use crate::utils::configs::extraction_config::Config;

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
        let image_data = STANDARD.decode(&png_page.base64_png)?;
        let part = reqwest::multipart::Part::bytes(image_data)
            .file_name(format!("{}.png", png_page.bb_id))
            .mime_str("image/png")?;
        form = form.part("images", part);
    }

    let response = client.post(url).multipart(form).send().await?;
    let response_body = response.text().await?;
    Ok(response_body)
}


pub async fn apply_llm_to_segments(
    segments: Vec<Segment>,
    llm_config: LLMConfig,
    png_pages: &[PngPage],
) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
    let table_prompt =
        "Extract all the tables from the image and return the data in a markdown table";
    let image_prompt = "Describe the image in detail";

    let extraction_config = Config::from_env()?;

    let mut result = Vec::new();
    for segment in segments {
        // Check if the segment type is in affected_segments
        if !llm_config.affected_segments.contains(&segment.segment_type) {
            result.push(segment);
            continue;
        }

        let (prompt, segment_type_str) = match segment.segment_type {
            SegmentType::Table => (table_prompt, "Table"),
            SegmentType::Picture => (image_prompt, "Image"),
            _ => continue, // Skip other segment types
        };

        let llm_output = apply_llm(
            llm_config.clone(),
            &extraction_config,
            png_pages.to_vec(),
            prompt.to_string(),
        )
        .await?;

        result.push(Segment {
            text: format!(
                "{}: {}\n\n LLM {} Output:\n{}",
                segment_type_str, segment.text, segment_type_str, llm_output
            ),
            ..segment
        });
    }
    Ok(result)
}
