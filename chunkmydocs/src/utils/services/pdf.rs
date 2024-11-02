use crate::utils::configs::extraction_config::Config as ExtractionConfig;
use crate::utils::configs::pdfium_config::Config as PdfiumConfig;
use image::ImageFormat;
use lopdf::Document;
use pdfium_render::prelude::*;
use std::{
    fs,
    path::{Path, PathBuf},
};
use uuid::Uuid;

pub fn split_pdf(
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

pub async fn pdf_2_images(
    pdf_path: &Path,
    temp_dir: &Path,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let config = PdfiumConfig::from_env()?;
    let dir_path = config.get_binary().await?;
    let extraction_config = ExtractionConfig::from_env()?;

    let pdfium = Pdfium::new(
        Pdfium::bind_to_system_library().or_else(|_| Pdfium::bind_to_library(&dir_path))?,
    );

    let document = pdfium.load_pdf_from_file(pdf_path, None)?;
    let render_config = PdfRenderConfig::new()
        .scale_page_by_factor(extraction_config.page_image_density / extraction_config.pdf_density);
    let mut image_paths = Vec::new();
    for (page_index, page) in document.pages().iter().enumerate() {
        let output_path = temp_dir.join(format!("page_{}.jpg", page_index + 1));

        page.render_with_config(&render_config)?
            .as_image()
            .into_rgb8()
            .save_with_format(output_path.clone(), ImageFormat::Jpeg)
            .map_err(|_| PdfiumError::ImageError)?;

        image_paths.push(output_path);
    }

    Ok(image_paths)
}
