use image::{imageops, DynamicImage, ImageBuffer, Rgba};
use imageproc::{
    contrast::adaptive_threshold, drawing::draw_hollow_rect_mut, filter::median_filter, rect::Rect,
};
use reqwest::multipart;
use std::{
    error::Error,
    fs::{self, File},
    io::BufWriter,
    path::Path,
};
use tempfile::NamedTempFile;

use crate::{
    models::workers::table_ocr::{TableStructure, TableStructureResponse},
    utils::configs::extraction_config::Config,
};

pub fn annotate_image(
    input_image_path: &Path,
    table_structures: &[TableStructure],
    output_folder: &Path,
) -> Result<(), Box<dyn Error>> {
    // Open the input image
    let mut img = image::open(input_image_path)?.to_rgba8();

    // Define the color for bounding boxes (e.g., red)
    let color = Rgba([255, 0, 0, 255]); // Red color with full opacity

    // Iterate over each table structure
    for table in table_structures {
        for cell in &table.cells {
            let bbox = &cell.cell;
            let rect = Rect::at(bbox.left as i32, bbox.top as i32)
                .of_size(bbox.width as u32, bbox.height as u32);

            // Draw the rectangle on the image
            draw_hollow_rect_mut(&mut img, rect, color);
        }
    }

    // Ensure the output directory exists
    fs::create_dir_all(output_folder)?;

    // Define the output image path
    let input_file_name = input_image_path
        .file_name()
        .ok_or("Invalid input file name")?;
    let output_image_path = output_folder.join(input_file_name);

    // Convert the image to RGB color space
    let rgb_img = DynamicImage::ImageRgba8(img).into_rgb8();

    // Save the annotated image as JPEG
    rgb_img.save_with_format(&output_image_path, image::ImageFormat::Jpeg)?;

    Ok(())
}
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

pub async fn recognize_table(
    file_path: &Path,
) -> Result<Vec<TableStructure>, Box<dyn Error + Send + Sync>> {
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
    use crate::utils::storage::config_s3::create_client;
    use crate::utils::storage::services::download_to_tempfile;
    use std::fs;
    use std::path::Path;
    use tokio;

    async fn setup_test_image(url: &str) -> std::path::PathBuf {
        let s3_client = create_client().await.unwrap();
        let reqwest_client = reqwest::Client::new();

        let temp_image = download_to_tempfile(&s3_client, &reqwest_client, url, None)
            .await
            .unwrap();

        let input_folder = Path::new("input");
        fs::create_dir_all(input_folder).unwrap();

        let input_path = input_folder.join("test_image.jpg");
        fs::copy(temp_image.path(), &input_path).unwrap();

        input_path
    }

    async fn run_table_recognition(image_path: &Path) -> Vec<TableStructure> {
        // let preprocessed_image = preprocess_image(image_path).await.unwrap();
        let result = recognize_table(image_path).await.unwrap();
        result
    }

    fn annotate_and_verify(input_path: &Path, tables: &[TableStructure]) {
        let output_folder = Path::new("output");
        fs::create_dir_all(output_folder).unwrap();

        let annotation_result = annotate_image(input_path, tables, output_folder);
        assert!(
            annotation_result.is_ok(),
            "annotate_image failed: {:?}",
            annotation_result.err()
        );
    }

    #[tokio::test]
    async fn test_recognize_table_from_url() {
        println!("Starting test_recognize_table_from_url");

        let url = "s3://chunkmydocs-bucket-prod/9e3c3093-b5da-4e8b-9d9f-239a596ce6fe/5c72edf7-ee0c-48cf-8ffd-273cb915855f/images/ee1c3808-4bdd-4305-baa3-d383fcc69838.jpg";
        let input_path = setup_test_image(url).await;
        println!("Image setup completed: {:?}", input_path);

        let tables = run_table_recognition(&input_path).await;
        println!("Table recognition completed");

        annotate_and_verify(&input_path, &tables);
        println!("Image annotation completed");

        println!("test_recognize_table_from_url completed successfully");
    }

    #[tokio::test]
    async fn test_recognize_table_from_local_file() {
        println!("Starting test_recognize_table_from_local_file");

        let input_path = Path::new("input/test_image.jpg");
        println!("Using local image: {:?}", input_path);

        let tables = run_table_recognition(input_path).await;
        println!("Table recognition completed");

        annotate_and_verify(input_path, &tables);
        println!("Image annotation completed");

        println!("test_recognize_table_from_local_file completed successfully");
    }
}
