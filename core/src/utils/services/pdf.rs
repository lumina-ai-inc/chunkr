use crate::utils::configs::pdfium_config::Config as PdfiumConfig;
use crate::utils::configs::worker_config::Config as WorkerConfig;
use image::ImageFormat;
use lopdf::Document;
use pdfium_render::prelude::*;
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;
use uuid::Uuid;

// Prob is not requried anymore
pub fn split_pdf(
    file_path: &Path,
    pages_per_split: usize,
    output_dir: &Path,
) -> Result<Vec<PathBuf>, Box<dyn Error>> {
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

pub fn pages_as_images(pdf_file: &NamedTempFile) -> Result<Vec<NamedTempFile>, Box<dyn Error>> {
    let extraction_config = WorkerConfig::from_env()?;
    let pdfium = PdfiumConfig::from_env()?.get_pdfium()?;
    let document = pdfium.load_pdf_from_file(pdf_file.path(), None)?;
    let render_config = PdfRenderConfig::new()
        .scale_page_by_factor(extraction_config.page_image_density / extraction_config.pdf_density);
    let mut image_files = Vec::new();
    for page in document.pages().iter() {
        let temp_file = NamedTempFile::new()?;

        page.render_with_config(&render_config)?
            .as_image()
            .into_rgb8()
            .save_with_format(temp_file.path(), ImageFormat::Jpeg)
            .map_err(|_| PdfiumError::ImageError)?;

        image_files.push(temp_file);
    }
    Ok(image_files)
}

pub fn extract_text(pdf_file: &NamedTempFile) -> Result<Vec<String>, Box<dyn Error>> {
    let pdfium = PdfiumConfig::from_env()?.get_pdfium()?;
    let document = pdfium.load_pdf_from_file(pdf_file.path(), None)?;
    let mut page_texts = Vec::new();
    for page in document.pages().iter() {
        let text = page.text().unwrap().all();
        page_texts.push(text);
    }

    Ok(page_texts)
}

pub fn count_pages(pdf_file: &NamedTempFile) -> Result<i32, Box<dyn std::error::Error>> {
    let pdfium = PdfiumConfig::from_env()?.get_pdfium()?;
    let document = pdfium.load_pdf_from_file(pdf_file.path(), None)?;
    Ok(document.pages().len() as i32)
}
