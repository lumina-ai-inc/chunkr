use crate::utils::configs::task_config::Config;
use lopdf::Document;
use reqwest::Client;
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

pub async fn convert_to_pdf(file_path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    let client = Client::new();
    let response = client
        .post(&format!("{}/to_pdf", config.service_url))
        .body(fs::read(file_path)?)
        .send()
        .await?;

    if response.status().is_success() {
        let temp_dir = tempfile::TempDir::new()?;
        let temp_file_path = temp_dir.path().join(file_path.file_name().unwrap());
        fs::write(&temp_file_path, response.bytes().await?)?;
        Ok(temp_file_path)
    } else {
        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to convert to PDF",
        )))
    }
}
