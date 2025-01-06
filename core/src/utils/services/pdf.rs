use crate::configs::pdfium_config::Config as PdfiumConfig;
use crate::models::chunkr::output::{BoundingBox, OCRResult};
use image::ImageFormat;
use pdfium_render::prelude::*;
use std::error::Error;
use tempfile::NamedTempFile;

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

pub fn count_pages(pdf_file: &NamedTempFile) -> Result<u32, Box<dyn std::error::Error>> {
    let pdfium = PdfiumConfig::from_env()?.get_pdfium()?;
    let document = pdfium.load_pdf_from_file(pdf_file.path(), None)?;
    Ok(document.pages().len() as u32)
}

pub fn extract_ocr_results(
    pdf_file: &NamedTempFile,
) -> Result<Vec<Vec<OCRResult>>, Box<dyn Error>> {
    let pdfium = PdfiumConfig::from_env()?.get_pdfium()?;
    let document = pdfium.load_pdf_from_file(pdf_file.path(), None)?;
    let mut page_results = Vec::new();

    for page in document.pages().iter() {
        let text_page = page.text()?;
        let mut page_ocr = Vec::new();

        for segment in text_page.segments().iter() {
            let rect = segment.bounds();
            page_ocr.push(OCRResult {
                bbox: BoundingBox {
                    left: rect.left.value,
                    top: rect.top.value,
                    width: rect.right.value - rect.left.value,
                    height: rect.bottom.value - rect.top.value,
                },
                text: segment.text(),
                confidence: None,
            });
        }

        page_results.push(page_ocr);
    }

    Ok(page_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::Path;

    #[test]
    fn test_count_pages() {
        let path = Path::new("input/test.pdf");
        let mut pdf_file = NamedTempFile::new().unwrap();
        pdf_file
            .write(std::fs::read(path).unwrap().as_slice())
            .unwrap();
        let count = count_pages(&pdf_file).unwrap();
        println!("Page count: {}", count);
    }

    #[test]
    fn test_extract_ocr_results() {
        let path = Path::new("input/test.pdf");
        let mut pdf_file = NamedTempFile::new().unwrap();
        pdf_file
            .write(std::fs::read(path).unwrap().as_slice())
            .unwrap();
        let ocr_results = extract_ocr_results(&pdf_file).unwrap();
        println!("OCR results: {:?}", ocr_results);
    }
}
