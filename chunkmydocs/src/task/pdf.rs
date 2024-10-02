use crate::utils::configs::task_config::Config;
use lopdf::Document;
use reqwest::{Client, multipart::{Form, Part}};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

pub async fn split_pdf(
    file_path: &Path,
    pages_per_split: usize,
    output_dir: &Path,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let doc = match Document::load(file_path) {
        Ok(doc) => doc,
        Err(e) => {
            eprintln!("Error loading PDF: {:?}", e);
            return Err(Box::new(e));
        }
    };
    let num_pages = doc.get_pages().len();

    fs::create_dir_all(output_dir)?;

    let mut split_files = Vec::new();

    for start_page in (1..=num_pages).step_by(pages_per_split) {
        let end_page = std::cmp::min(start_page + pages_per_split - 1, num_pages);

        let mut batch_doc = doc.clone();

        let pages_to_delete: Vec<u32> = (1..=num_pages as u32)
            .filter(|&page| (page < (start_page as u32) || page > (end_page as u32)))
            .collect();

        batch_doc.delete_pages(&pages_to_delete);

        let filename = format!("{}.pdf", Uuid::new_v4());
        let file_path = output_dir.join(filename);

        batch_doc.save(&file_path)?;

        split_files.push(file_path);
    }

    Ok(split_files)
}


pub async fn convert_to_pdf(
    input_file_path: &Path,
    output_file_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    let client = Client::new();

    let url = format!("{}/to_pdf", config.service_url);

    let file_name = input_file_path
        .file_name()
        .ok_or_else(|| format!("Invalid file name: {:?}", input_file_path))?
        .to_str()
        .ok_or_else(|| format!("Non-UTF8 file name: {:?}", input_file_path))?
        .to_string();

    let file_fs = fs::read(input_file_path)?;
    let part = Part::bytes(file_fs).file_name(file_name);

    let form = Form::new().part("file", part);

    let response = client.post(&url).multipart(form).send().await?;


    if response.status().is_success() {
        let content = response.bytes().await?;
        fs::write(output_file_path, &content)?;
        Ok(())
    } else {
        let status = response.status();
        let error_message = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        println!(
            "Failed to convert to PDF for file: {:?}, Status: {}, Message: {}",
            input_file_path, status, error_message
        );
        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "Failed to convert to PDF. Status: {}, Message: {}",
                status, error_message
            ),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;
    use tokio;

    #[tokio::test]
    async fn test_convert_to_pdf() {
        // Use the test.docx file from the input folder
        let input_path = PathBuf::from("input").join("test.docx");

        // Create a temporary file for the output
        let output_file = NamedTempFile::new().unwrap();
        let output_path = output_file.path().to_path_buf();

        // Call the convert_to_pdf function
        let result = convert_to_pdf(&input_path, &output_path).await;

        // Check the result
        match result {
            Ok(_) => {
                let output_dir = PathBuf::from("output");
                fs::create_dir_all(&output_dir).unwrap();
                let final_output_path = output_dir.join("test_output.pdf");
                fs::copy(&output_path, &final_output_path).unwrap();
                println!("Test output saved to {:?}", final_output_path);
                assert!(final_output_path.exists());
            }
            Err(e) => {
                panic!("PDF conversion failed: {:?}", e);
            }
        }
    }
}
