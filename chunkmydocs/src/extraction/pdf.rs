use lopdf::Document;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

// use crate::models::extraction::segment::Segment;
// use image::{DynamicImage, RgbaImage};
// use poppler::Document as PopplerDocument;
// use poppler::Page as PopplerPage;
// use poppler::{Document as PopplerDocument, Page as PopplerPage};

pub fn split_pdf(
    file_path: &Path,
    pages_per_split: usize,
    output_dir: &Path,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    // Load the document
    let doc = Document::load(file_path)?;
    let num_pages = doc.get_pages().len();

    // Create the output directory if it doesn't exist
    fs::create_dir_all(output_dir)?;

    let mut split_files = Vec::new();

    for start_page in (1..=num_pages).step_by(pages_per_split) {
        let end_page = std::cmp::min(start_page + pages_per_split - 1, num_pages);

        // Create a clone of the document for this batch
        let mut batch_doc = doc.clone();

        // Calculate pages to delete
        let pages_to_delete: Vec<u32> = (1..=num_pages as u32)
            .filter(|&page| (page < (start_page as u32) || page > (end_page as u32)))
            .collect();

        // Delete pages not in this batch
        batch_doc.delete_pages(&pages_to_delete);

        // Generate a unique filename using UUID
        let filename = format!("{}.pdf", Uuid::new_v4());
        let file_path = output_dir.join(filename);

        // Save the batch document to the file
        batch_doc.save(&file_path)?;

        // Add the file path to our vector
        split_files.push(file_path);
    }

    println!("Split files: {:?}", split_files);

    Ok(split_files)
}

// pub fn extract_segment_images(
//     pdf_file_path: &Path,
//     json_file_path: &Path,
//     output_dir: &Path,
// ) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
//     // Load the PDF document
//     let doc = Document::load(pdf_file_path)?;

//     // Read and parse the JSON file
//     let json_content = std::fs::read_to_string(json_file_path)?;
//     let segments: Vec<Segment> = serde_json::from_str(&json_content)?;

//     // Create the output directory if it doesn't exist
//     std::fs::create_dir_all(output_dir)?;

//     let mut extracted_images = Vec::new();

//     for (index, segment) in segments.iter().enumerate() {
//         // Convert PDF page to image
//         let mut page_image = doc_to_image(&doc, segment.page_number)?;

//         // Extract segment area
//         let x = (segment.left * page_image.width() as f32) as u32;
//         let y = (segment.top * page_image.height() as f32) as u32;
//         let width = (segment.width * page_image.width() as f32) as u32;
//         let height = (segment.height * page_image.height() as f32) as u32;

//         let segment_image = page_image.crop(x, y, width, height);

//         // Generate a unique filename
//         let filename = format!("segment_{}.png", index);
//         let file_path = output_dir.join(filename);

//         // Save the segment image
//         segment_image.save(&file_path)?;
//         extracted_images.push(file_path);
//     }

//     Ok(extracted_images)
// }

// fn doc_to_image(
//     doc: &Document,
//     page_number: u32,
// ) -> Result<DynamicImage, Box<dyn std::error::Error>> {
//     // Convert lopdf::Document to poppler::Document
//     let pdf_data = doc.save_to_bytes()?;
//     let poppler_doc = PopplerDocument::from_data(&pdf_data, None)?;

//     // Page numbers in Poppler are 0-indexed
//     let page = poppler_doc
//         .page(page_number as i32 - 1)
//         .ok_or("Page not found")?;

//     // Get page dimensions
//     let (width, height) = page.size();

//     // Render page to image
//     let mut img = RgbaImage::new(width as u32, height as u32);
//     page.render(
//         img.as_mut(),
//         poppler::PopplerRectangle {
//             x1: 0.0,
//             y1: 0.0,
//             x2: width,
//             y2: height,
//         },
//         poppler::CairoScale { x: 1.0, y: 1.0 },
//         poppler::Rotation::Upright,
//     );

//     Ok(DynamicImage::ImageRgba8(img))
// }
