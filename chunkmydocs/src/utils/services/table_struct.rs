use crate::models::workers::table_ocr::{TableStructure, TableStructureResponse};
use crate::utils::configs::extraction_config::Config;
use image::imageops;
use image::{ImageBuffer, Luma};
use imageproc::contrast::adaptive_threshold;
use imageproc::filter::median_filter;
use reqwest::multipart;
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::{fs, path::Path};
use tempfile::NamedTempFile;
pub async fn preprocess_image(
    input_path: &std::path::Path,
) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    let img = image::open(input_path)?;
    let gray_img = img.to_luma8();
    let enhanced_img = imageops::contrast(&gray_img, 1.1); // Reduced contrast enhancement further
    let denoised_img = median_filter(&enhanced_img, 3, 3); // Kept the same

    let binary_img = adaptive_threshold(
        &ImageBuffer::from_raw(
            denoised_img.width(),
            denoised_img.height(),
            denoised_img.clone().into_raw(),
        )
        .unwrap(),
        61, // Further increased threshold window size for even less aggressive binarization
    );

    let temp_file = NamedTempFile::new()?;
    let file = File::create(temp_file.path())?;
    let mut w = BufWriter::new(file);
    binary_img.write_to(&mut w, image::ImageFormat::Png)?; // Write denoised image instead of binary

    Ok(temp_file)
}

pub async fn recognize_table(file_path: &Path) -> Result<Vec<TableStructure>, Box<dyn Error>> {
    let client = reqwest::Client::new();
    let config = Config::from_env()?;
    let url = format!("{}/predict/table", &config.table_structure_url);

    let file_name = file_path
        .file_name()
        .ok_or_else(|| format!("Invalid file name: {:?}", file_path))?
        .to_str()
        .ok_or_else(|| format!("Non-UTF8 file name: {:?}", file_path))?
        .to_string();

    let file_fs = fs::read(file_path).expect("Failed to read file");
    let part = multipart::Part::bytes(file_fs).file_name(file_name);
    let form = multipart::Form::new().part("files", part);
    let response = client
        .post(&url)
        .multipart(form)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await?
        .error_for_status()?;

    let table_struct_response: TableStructureResponse = response.json().await?;
    Ok(table_struct_response.result)
}
#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_recognize_table() {
        // Path to the test image
        let image_path = std::path::Path::new(
            "/Users/ishaankapoor/Startup/chunk-my-docs/chunkmydocs/input/test.jpg",
        );
        let preprocessed_image = preprocess_image(image_path).await;
        // Call the recognize_table function
        let result = recognize_table(preprocessed_image.unwrap().path()).await;

        // Assert that the result is Ok
        assert!(result.is_ok(), "recognize_table failed: {:?}", result.err());
    }
}
