use lopdf::Document;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;
use pdfium_render::prelude::*;

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

pub fn pdf_2_images(pdf_path: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let pdfium = Pdfium::new(
        Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(".")).unwrap()
    );

    let document = pdfium.load_pdf_from_file(pdf_path, None)?;
    let mut image_paths = Vec::new();
    for (page_index, page) in document.pages().iter().enumerate() {
        let bitmap = page.render_with_config(&PdfRenderConfig::new()
            .set_target_width(2000)
            .set_maximum_height(3000) 
        )?;

        let output_path = format!("page_{}.png", page_index + 1);
        bitmap.as_image().save(Path::new(&output_path))?;
        image_paths.push(PathBuf::from(output_path));
    }

    Ok(image_paths)
}