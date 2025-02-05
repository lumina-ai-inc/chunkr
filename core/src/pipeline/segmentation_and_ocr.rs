use crate::models::chunkr::output::{BoundingBox, Chunk, OCRResult, Segment, SegmentType};
use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::{Status, Task};
use crate::models::chunkr::upload::{OcrStrategy, SegmentationStrategy};
use crate::utils::services::images;
use crate::utils::services::ocr;
use crate::utils::services::pdf;
use crate::utils::services::segmentation;
use tempfile::NamedTempFile;

async fn page_segmentation(
    page: &NamedTempFile,
    ocr_results: Vec<OCRResult>,
    page_number: u32,
) -> Result<Vec<Segment>, Box<dyn std::error::Error + Send + Sync>> {
    let (page_width, page_height) = images::get_image_dimensions(page)?;
    let segments = vec![Segment::new(
        BoundingBox::new(0.0, 0.0, page_width as f32, page_height as f32),
        Some(1.0),
        ocr_results,
        page_height as f32,
        page_width as f32,
        page_number,
        SegmentType::Page,
    )];
    Ok(segments)
}

async fn ocr_pages_batch(
    pages: &[&NamedTempFile],
) -> Result<Vec<Vec<OCRResult>>, Box<dyn std::error::Error + Send + Sync>> {
    Ok(ocr::perform_general_ocr_batch(pages).await?)
}

pub async fn process_segmentation(task: &mut Task, pages: &[&NamedTempFile], ocr_results: Vec<Vec<OCRResult>>) -> Result<Vec<Vec<Segment>>, Box<dyn std::error::Error + Send + Sync>> {
    let configuration = task.configuration.clone();
    match configuration.segmentation_strategy {
            SegmentationStrategy::LayoutAnalysis => {
                segmentation::perform_segmentation_batch(&pages, ocr_results).await
            }
            SegmentationStrategy::Page => {
                let mut segments = Vec::new();
                for (idx, (page, ocr)) in pages.iter().zip(ocr_results.into_iter()).enumerate() {
                    segments.push(page_segmentation(page, ocr, idx as u32 + 1).await?);
                }
                Ok(segments)
            }
        }
}

async fn process_ocr(task: &mut Task, pdf_file: &NamedTempFile, scaling_factor: f32, pages: &[&NamedTempFile]) -> Result<Vec<Vec<OCRResult>>, Box<dyn std::error::Error + Send + Sync>> {
    let configuration = task.configuration.clone();
    let pdf_ocr_results =
    match pdf::extract_ocr_results(pdf_file, scaling_factor) {
        Ok(ocr_results) => ocr_results,
        Err(e) => {
            println!("Error getting pdf ocr results: {:?}", e);
            vec![vec![]; task.page_count.unwrap_or(0) as usize]
        }
    };
    match configuration.ocr_strategy {
        OcrStrategy::All => Ok(ocr_pages_batch(&pages).await?),
        OcrStrategy::Auto => {
            if pdf_ocr_results.iter().all(|r| r.is_empty()) {
                Ok(ocr_pages_batch(&pages).await?)
            } else {
                Ok(pdf_ocr_results)
            }
        }
    }
}

/// Process the pages
///
/// This function will perform OCR, segmentation and chunking on the pages
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn std::error::Error>> {
    let mut task = pipeline.get_task()?;
    let pdf_file = pipeline.pdf_file.as_ref().unwrap();
    let scaling_factor = task.configuration.get_scaling_factor()?;
    let pages: Vec<_> = pipeline
        .page_images
        .as_ref()
        .unwrap()
        .iter()
        .map(|x| x.as_ref())
        .collect();

    task.update(
        Some(Status::Processing),
        Some("Performing OCR".to_string()),
        None,
        None,
        None,
            None,
            None,
        )
        .await?;

    let ocr_results = match process_ocr(&mut task, pdf_file, scaling_factor, &pages).await {
        Ok(ocr_results) => ocr_results,
        Err(e) => {
            println!("Error in OCR: {:?}", e);
            return Err(e.to_string().into());
        }
    };

    task.update(
        Some(Status::Processing),
        Some("Performing Segmentation".to_string()),
        None,
        None,
        None,
            None,
            None,
        )
        .await?;

    let page_segments = match process_segmentation(&mut task, &pages, ocr_results).await {
        Ok(page_segments) => page_segments,
        Err(e) => {
            println!("Error in segmentation and OCR: {:?}", e);
            return Err(e.to_string().into());
        }
    };

    pipeline.chunks = page_segments
        .into_iter()
        .flatten()
        .map(|s| Chunk::new(vec![s]))
        .collect();

    Ok(())
}
