use crate::configs::pdfium_config::Config as PdfiumConfig;
use crate::models::output::{BoundingBox, OCRResult};
use image::ImageFormat;
use image::ImageReader;
use pdfium_render::prelude::*;
use std::error::Error;
use tempfile::{Builder, NamedTempFile};

/// Render each PDF page to a JPEG and return a Vec of temp files.
pub fn pages_as_images(
    pdf_file: &NamedTempFile,
    scaling_factor: f32,
) -> Result<Vec<NamedTempFile>, Box<dyn Error>> {
    let pdfium = PdfiumConfig::from_env()?.get_pdfium()?;
    let document = pdfium.load_pdf_from_file(pdf_file.path(), None)?;
    let render_config = PdfRenderConfig::new().scale_page_by_factor(scaling_factor);

    let page_count = document.pages().len();
    let mut image_files = Vec::with_capacity(page_count.into());

    document.pages().iter().try_for_each(|page| {
        let temp_file = NamedTempFile::new()?;
        page.render_with_config(&render_config)?
            .as_image()
            .into_rgb8()
            .save_with_format(temp_file.path(), ImageFormat::Jpeg)
            .map_err(|_| PdfiumError::ImageError)?;
        image_files.push(temp_file);
        Ok::<_, Box<dyn Error>>(())
    })?;

    Ok(image_files)
}

/// Count the number of pages in the PDF.
pub fn count_pages(pdf_file: &NamedTempFile) -> Result<u32, Box<dyn std::error::Error>> {
    let pdfium = PdfiumConfig::from_env()?.get_pdfium()?;
    let document = pdfium.load_pdf_from_file(pdf_file.path(), None)?;
    Ok(document.pages().len() as u32)
}

/// Extracts OCR results from each PDF page by grouping individual characters
/// into words (splitting on actual whitespace), and computing each word's bounding box
/// as the union of its character boxes. Converts from bottom-left to top-left origin.
pub fn extract_ocr_results(
    pdf_file: &NamedTempFile,
    scaling_factor: f32,
) -> Result<Vec<Vec<OCRResult>>, Box<dyn Error>> {
    let pdfium = PdfiumConfig::from_env()?.get_pdfium()?;
    let document = pdfium.load_pdf_from_file(pdf_file.path(), None)?;
    let mut all_pages = Vec::with_capacity(document.pages().len().into());

    for page in document.pages().iter() {
        let text_page = page.text()?;
        let page_height = page.height().value;
        let mut page_results = Vec::new();
        let mut current_chars: Vec<(char, PdfRect)> = Vec::new();

        for text_char in text_page.chars().iter() {
            // Get the Unicode character (if any) and its tight bounding box
            let c_opt = text_char.unicode_char();
            let rect = text_char.tight_bounds()?;

            match c_opt {
                Some(c) if !c.is_whitespace() => {
                    // Part of a word
                    current_chars.push((c, rect));
                }
                _ => {
                    // Whitespace or unrecognized: flush any accumulated word
                    if !current_chars.is_empty() {
                        page_results.push(build_word(&current_chars, page_height, scaling_factor));
                        current_chars.clear();
                    }
                }
            }
        }

        // Flush the final word on the page, if any
        if !current_chars.is_empty() {
            page_results.push(build_word(&current_chars, page_height, scaling_factor));
        }

        all_pages.push(page_results);
    }

    Ok(all_pages)
}

/// Helper: given a slice of (char, PdfRect) for one word, build an OCRResult
/// whose bbox is the union of all character rectangles, and whose text is
/// the concatenation of the chars.
fn build_word(chars: &[(char, PdfRect)], page_height: f32, scale: f32) -> OCRResult {
    let mut text = String::new();
    let mut min_left = f32::INFINITY;
    let mut max_right = f32::NEG_INFINITY;
    let mut min_top = f32::INFINITY;
    let mut max_bottom = f32::NEG_INFINITY;

    for (c, rect) in chars.iter() {
        text.push(*c);
        min_left = min_left.min(rect.left.value);
        max_right = max_right.max(rect.right.value);
        max_bottom = max_bottom.max(rect.bottom.value);
        min_top = min_top.min(rect.top.value);
    }

    let width_pdf = max_right - min_left;
    let height_pdf = (min_top - max_bottom).abs();

    OCRResult {
        bbox: BoundingBox {
            left: min_left * scale,
            top: (page_height - min_top) * scale,
            width: width_pdf * scale,
            height: height_pdf * scale,
        },
        text,
        confidence: None,
    }
}

/// Combines multiple PDFs into a single PDF by appending all pages.
/// Takes a list of PDF files as NamedTempFile and returns a new NamedTempFile
/// containing the combined PDF.
pub fn combine_pdfs(pdf_files: Vec<&NamedTempFile>) -> Result<NamedTempFile, Box<dyn Error>> {
    let pdfium = PdfiumConfig::from_env()?.get_pdfium()?;

    // Create a new blank document
    let mut combined_document = pdfium.create_new_pdf()?;

    // Append all pages from each PDF file
    for pdf_file in pdf_files {
        let source_document = pdfium.load_pdf_from_file(pdf_file.path(), None)?;
        combined_document.pages_mut().append(&source_document)?;
    }

    // Create a temporary file for the output
    let output_file = NamedTempFile::new()?;

    // Save the combined document to the temporary file
    combined_document.save_to_file(output_file.path())?;

    Ok(output_file)
}

/// Convert an image file to a PDF with the image filling a single page.
/// Creates a PDF page with the same dimensions as the image and places the image in it.
pub fn create_pdf_from_image(
    input_file: &NamedTempFile,
) -> Result<NamedTempFile, Box<dyn Error + Send + Sync>> {
    let output_file = Builder::new().suffix(".pdf").tempfile()?;
    let img = ImageReader::open(input_file.path())?
        .with_guessed_format()?
        .decode()?;
    let width = img.width();
    let height = img.height();
    let pdfium = PdfiumConfig::from_env()?
        .get_pdfium()
        .map_err(|e| Box::<dyn Error + Send + Sync>::from(e.to_string()))?;
    let mut document = pdfium.create_new_pdf()?;
    let page_width = PdfPoints::new(width as f32 * 72.0);
    let page_height = PdfPoints::new(height as f32 * 72.0);
    let mut page = document
        .pages_mut()
        .create_page_at_end(PdfPagePaperSize::Custom(page_width, page_height))?;
    page.objects_mut().create_image_object(
        PdfPoints::ZERO,
        PdfPoints::ZERO,
        &img,
        Some(page_width),
        Some(page_height),
    )?;
    document.save_to_file(output_file.path())?;
    Ok(output_file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use tempfile::Builder;

    #[test]
    fn test_create_pdf_from_image() {
        let input_file_path = Path::new("/home/akhilesh/Downloads/page_3.jpg");
        let mut output_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        output_file_path.push("output/pdf/test_create_pdf_from_image.pdf");
        std::fs::create_dir_all(output_file_path.parent().unwrap()).unwrap();
        let input_file = Builder::new().suffix(".jpg").tempfile().unwrap();
        fs::copy(input_file_path, input_file.path()).unwrap();
        let output_file = create_pdf_from_image(&input_file).unwrap();
        fs::copy(output_file.path(), &output_file_path).unwrap();
        assert!(output_file.path().exists());
        println!("Output file path: {}", output_file_path.as_path().display());
    }
}
