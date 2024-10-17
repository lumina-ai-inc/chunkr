use crate::models::server::segment::OCRResult;
use crate::utils::services::rapid_ocr::call_rapid_ocr_api;
use crate::utils::services::table_struct::recognize_table;
use crate::utils::storage::services::download_to_tempfile;
use aws_sdk_s3::Client as S3Client;
use reqwest::Client as ReqwestClient;

use image::{ImageBuffer, Luma};
use imageproc::contrast::adaptive_threshold;
use std::fs::File;
use std::io::BufWriter;
use tempfile::NamedTempFile;

async fn binarize_image(
    input_path: &std::path::Path,
) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    // Open the image
    let img = image::open(input_path)?;

    // Convert to grayscale
    let gray_img = img.to_luma8();

    // Binarize the image
    let binary_img: ImageBuffer<Luma<u8>, Vec<u8>> = adaptive_threshold(&gray_img, 128);

    // Create a temporary file for the binarized image
    let temp_file = NamedTempFile::new()?;

    // Save the binarized image to the temporary file
    let file = File::create(temp_file.path())?;
    let mut w = BufWriter::new(file);
    binary_img.write_to(&mut w, image::ImageFormat::Png)?;

    Ok(temp_file)
}

pub async fn download_and_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str,
) -> Result<Vec<OCRResult>, Box<dyn std::error::Error>> {
    println!("Downloading and OCRing image: {:?}", image_location);
    let temp_file = download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;

    // Binarize the downloaded image
    let binarized_temp_file = binarize_image(temp_file.path()).await?;

    let ocr_results = call_rapid_ocr_api(binarized_temp_file.path()).await?;
    let table_structures = recognize_table(binarized_temp_file.path()).await?;

    // Clean up temporary files
    temp_file.close()?;
    binarized_temp_file.close()?;

    Ok(ocr_results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_binarize_image() {
        // Get the current directory
        let current_dir = std::env::current_dir().unwrap();

        // Construct the path to the test image
        let image_path = current_dir.join("input").join("test.jpg");

        // Ensure the test image exists
        assert!(
            image_path.exists(),
            "Test image does not exist at {:?}",
            image_path
        );

        // Binarize the image
        let binarized_temp_file = binarize_image(&image_path).await.unwrap();
        // Save the binarized image to the output folder
        let output_dir = current_dir.join("output");
        std::fs::create_dir_all(&output_dir).unwrap();
        let output_path = output_dir.join("binarized_test.png");
        std::fs::copy(binarized_temp_file.path(), &output_path).unwrap();

        println!("Binarized image saved to: {:?}", output_path);

        // Open the binarized image
        assert!(output_path.exists());
        // Check that the image only contains black (0) or white (255) pixels
    }
}
