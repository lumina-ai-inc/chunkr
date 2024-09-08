use crate::models::server::segment::{Chunk, Segment};
use crate::utils::configs::pdf2png_config::Config;
use reqwest::{multipart, Client as ReqwestClient};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::sync::OnceCell;

static REQWEST_CLIENT: OnceCell<ReqwestClient> = OnceCell::const_new();

#[derive(Serialize, Deserialize, Clone)]
pub struct BoundingBox {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    pub page_number: u32,
    pub bb_id: String,
}

impl From<&Segment> for BoundingBox {
    fn from(segment: &Segment) -> Self {
        BoundingBox {
            left: segment.left / segment.page_width,
            top: segment.top / segment.page_height,
            width: segment.width / segment.page_width,
            height: segment.height / segment.page_height,
            page_number: segment.page_number,
            bb_id: uuid::Uuid::new_v4().to_string(),
        }
    }
}

async fn get_reqwest_client() -> &'static ReqwestClient {
    REQWEST_CLIENT
        .get_or_init(|| async { ReqwestClient::new() })
        .await
}

pub async fn convert_pdf_to_png(
    pdf_path: &Path,
    bounding_boxes: Vec<BoundingBox>,
) -> Result<ConversionResponse, Box<dyn std::error::Error>> {
    // Updated return type
    let client = get_reqwest_client().await;
    let config = Config::from_env()?;
    let url = config.url;

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

    let response_json: ConversionResponse = response.json().await?; // Parse response as ConversionResponse

    Ok(response_json)
}

// Define the ConversionResponse struct
#[derive(Serialize, Deserialize)]
pub struct ConversionResponse {
    pub png_pages: Vec<PngPage>,
}

#[derive(Serialize, Deserialize)]
pub struct PngPage {
    pub bb_id: String,
    pub base64_png: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64;
    use serde_json::Value;
    use std::fs;
    use std::path::PathBuf;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_png() -> Result<(), Box<dyn std::error::Error>> {
        // Define input and output paths
        let input_dir = PathBuf::from("input");
        let output_dir = PathBuf::from("output");
        fs::create_dir_all(&input_dir)?;
        fs::create_dir_all(&output_dir)?;

        // Define the PDF file path
        let pdf_file_path = input_dir.join("test.pdf");

        // Load and parse the JSON file
        let json_content = fs::read_to_string(input_dir.join("test.json"))?;
        let json: Value = serde_json::from_str(&json_content)?;

        // Extract bounding boxes for "Table" and "Picture" types
        let bounding_boxes: Vec<BoundingBox> = json
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .flat_map(|page| {
                let segments: Vec<_> = page["segments"].as_array().unwrap_or(&Vec::new()).clone();
                segments
                    .into_iter()
                    .filter(|segment| {
                        let segment_type = segment["type"].as_str().unwrap_or("");
                        segment_type == "Table"
                    })
                    .map(|segment| {
                        let page_width = segment["page_width"].as_f64().unwrap_or(1.0);
                        let page_height = segment["page_height"].as_f64().unwrap_or(1.0);
                        BoundingBox {
                            left: segment["left"].as_f64().unwrap_or(0.0) as f32,
                            top: segment["top"].as_f64().unwrap_or(0.0) as f32,
                            width: segment["width"].as_f64().unwrap_or(0.0) as f32,
                            height: segment["height"].as_f64().unwrap_or(0.0) as f32,
                            page_number: segment["page_number"].as_i64().unwrap_or(1) as u32,
                            bb_id: Uuid::new_v4().to_string(),
                        }
                    })
            })
            .collect();

        // Convert PDF to PNG
        let response = convert_pdf_to_png(&pdf_file_path, bounding_boxes).await?;

        // Save the full JSON response
        let response_path = output_dir.join("response.json");
        fs::write(&response_path, serde_json::to_string_pretty(&response)?)?;

        // Save PNG files
        for png_page in &response.png_pages {
            let png_data = base64::decode(&png_page.base64_png)?;
            let png_path = output_dir.join(format!("snip_{}.png", png_page.bb_id));
            fs::write(&png_path, png_data)?;
        }

        println!("PDF conversion test passed successfully.");
        println!("Output saved in {:?}", output_dir);

        Ok(())
    }
}
