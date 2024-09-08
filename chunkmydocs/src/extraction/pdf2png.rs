use crate::models::server::segment::Segment;
use crate::utils::configs::pdf2png_config::Config;
use reqwest::{multipart, Client as ReqwestClient};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::sync::OnceCell;

static REQWEST_CLIENT: OnceCell<ReqwestClient> = OnceCell::const_new();
use base64::{engine::general_purpose::STANDARD, Engine as _};

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
            left: segment.left,
            top: segment.top,
            width: segment.width,
            height: segment.height,
            page_number: segment.page_number,
            bb_id: uuid::Uuid::new_v4().to_string(),
        }
    }
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

#[derive(Deserialize)]
pub struct SplitPdf {
    pub split_number: usize,
    pub base64_pdf: String,
}

#[derive(Deserialize)]
pub struct SplitResponse {
    pub split_pdfs: Vec<SplitPdf>,
}

async fn get_reqwest_client() -> &'static ReqwestClient {
    REQWEST_CLIENT
        .get_or_init(|| async { ReqwestClient::new() })
        .await
}

pub async fn split_pdf(
    file_path: &Path,
    pages_per_split: usize,
    output_dir: &Path,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let client = get_reqwest_client().await;
    let config = Config::from_env()?;
    let url = format!("{}/split", config.url);

    // Create the output directory if it doesn't exist
    tokio::fs::create_dir_all(output_dir).await?;

    let file_name = file_path
        .file_name()
        .ok_or_else(|| "Invalid file name")?
        .to_str()
        .ok_or_else(|| "Non-UTF8 file name")?;

    let file_content = tokio::fs::read(file_path).await?;
    let part = multipart::Part::bytes(file_content).file_name(file_name.to_string());

    let form = multipart::Form::new()
        .part("file", part)
        .text("pages_per_split", pages_per_split.to_string());

    let response = client
        .post(url)
        .multipart(form)
        .send()
        .await?
        .error_for_status()?;

    let split_response: SplitResponse = response.json().await?;
    let mut split_files = Vec::new();

    for split_pdf in split_response.split_pdfs.iter() {
        let pdf_data = STANDARD.decode(&split_pdf.base64_pdf)?;
        let filename = format!("split_{}.pdf", split_pdf.split_number);
        let file_path = output_dir.join(&filename);

        tokio::fs::write(&file_path, pdf_data).await?;
        split_files.push(file_path);
    }

    Ok(split_files)
}

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

    Ok(response_json)
}

#[cfg(test)]
mod tests {
    use super::*;
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
            let png_data = STANDARD.decode(&png_page.base64_png)?;
            let png_path = output_dir.join(format!("snip_{}.png", png_page.bb_id));
            fs::write(&png_path, png_data)?;
        }

        println!("PDF conversion test passed successfully.");
        println!("Output saved in {:?}", output_dir);

        Ok(())
    }
}
