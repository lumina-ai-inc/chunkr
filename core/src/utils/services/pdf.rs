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

/// Extracts OCR results from a PDF file.
///
/// The OCR results are converted from a bottom-left origin to a top-left origin.
pub fn extract_ocr_results(
    pdf_file: &NamedTempFile,
    scaling_factor: f32,
) -> Result<Vec<Vec<OCRResult>>, Box<dyn Error>> {
    let pdfium = PdfiumConfig::from_env()?.get_pdfium()?;
    let document = pdfium.load_pdf_from_file(pdf_file.path(), None)?;
    let mut page_results = Vec::new();

    for page in document.pages().iter() {
        let text_page = page.text()?;
        let page_height = page.height().value;
        let mut page_ocr = Vec::new();
        for segment in text_page.segments().iter() {
            let rect = segment.bounds();
            let ocr_result = OCRResult {
                bbox: BoundingBox {
                    left: rect.left.value * scaling_factor,
                    top: (page_height - rect.top.value) * scaling_factor,
                    width: (rect.right.value - rect.left.value) * scaling_factor,
                    height: (rect.top.value - rect.bottom.value).abs() * scaling_factor,
                },
                text: segment.text(),
                confidence: None,
            };
            page_ocr.push(ocr_result);
        }
        page_results.push(page_ocr);
    }

    Ok(page_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    use imageproc::drawing::draw_hollow_rect_mut;
    use imageproc::rect::Rect;
    use std::io::Write;
    use std::path::Path;

    #[test]
    fn test_visualize_ocr_boxes() -> Result<(), Box<dyn Error>> {
        let path = Path::new("input/test.pdf");
        let output_dir = Path::new("output/annotated_pages");
        let mut pdf_file = NamedTempFile::new()?;
        pdf_file.write(std::fs::read(path)?.as_slice())?;

        let ocr_results = extract_ocr_results(&pdf_file, 1.0)?;
        let image_files = pages_as_images(&pdf_file, 1.0)?;

        std::fs::create_dir_all(output_dir)?;

        for (page_idx, (image_file, page_ocr)) in
            image_files.iter().zip(ocr_results.iter()).enumerate()
        {
            let mut img = image::io::Reader::open(image_file.path())?
                .with_guessed_format()?
                .decode()?
                .into_rgb8();

            for ocr in page_ocr {
                let rect = Rect::at(ocr.bbox.left.round() as i32, ocr.bbox.top.round() as i32)
                    .of_size(
                        ocr.bbox.width.round() as u32,
                        ocr.bbox.height.round() as u32,
                    );
                draw_hollow_rect_mut(&mut img, rect, Rgb([255, 0, 0]));
            }

            img.save(format!("{}/page_{}.jpg", output_dir.display(), page_idx))?;
        }

        Ok(())
    }

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
        let ocr_results = extract_ocr_results(&pdf_file, 1.0).unwrap();
        println!("OCR results: {:?}", ocr_results);
    }
}
