use crate::models::chunkr::output::{BoundingBox, OCRResult, Segment, SegmentType};
use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::{Status, TaskPayload};
use crate::models::chunkr::upload::{OcrStrategy, SegmentationStrategy};
use crate::utils::services::chunking;
use crate::utils::services::images;
use crate::utils::services::ocr;
use crate::utils::services::pdf;
use crate::utils::services::segmentation;
use futures::future::try_join_all;
use rayon::prelude::*;
use tempfile::NamedTempFile;

async fn ocr_page_all(page: &NamedTempFile) -> Result<Vec<OCRResult>, Box<dyn std::error::Error>> {
    match ocr::perform_general_ocr(&page).await {
        Ok(ocr_results) => Ok(ocr_results),
        Err(e) => {
            println!("Error in performing OCR: {:?}", e);
            Err(e.to_string().into())
        }
    }
}

async fn ocr_page_auto(
    page: &NamedTempFile,
    extracted_ocr_result: Vec<OCRResult>,
) -> Result<Vec<OCRResult>, Box<dyn std::error::Error>> {
    if extracted_ocr_result.is_empty() {
        match ocr::perform_general_ocr(&page).await {
            Ok(ocr_results) => Ok(ocr_results),
            Err(e) => {
                println!("Error in performing OCR: {:?}", e);
                Err(e.to_string().into())
            }
        }
    } else {
        Ok(extracted_ocr_result)
    }
}

async fn page_segmentation(
    page: &NamedTempFile,
    ocr_results: Vec<OCRResult>,
    page_number: u32,
) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
    let (page_width, page_height) = images::get_image_dimensions(page)?;
    let segments = vec![Segment::new(
        BoundingBox::new(0.0, 0.0, page_width as f32, page_height as f32),
        ocr_results,
        page_height as f32,
        page_number,
        page_width as f32,
        SegmentType::Page,
    )];
    Ok(segments)
}

async fn layout_analysis(
    page: &NamedTempFile,
    ocr_results: Vec<OCRResult>,
    page_number: u32,
) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
    let segments = match segmentation::perform_segmentation(page, ocr_results, page_number).await {
        Ok(segments) => segments,
        Err(e) => {
            println!("Error in performing segmentation: {:?}", e);
            return Err(e.to_string().into());
        }
    };
    Ok(segments)
}

async fn process_page(
    page: &NamedTempFile,
    task_payload: TaskPayload,
    extracted_ocr_result: Vec<OCRResult>,
    page_number: u32,
) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
    let ocr_results = match task_payload.current_configuration.ocr_strategy {
        OcrStrategy::All => ocr_page_all(&page).await?,
        OcrStrategy::Auto => ocr_page_auto(&page, extracted_ocr_result).await?,
    };
    let segments = match task_payload.current_configuration.segmentation_strategy {
        SegmentationStrategy::LayoutAnalysis => {
            layout_analysis(page, ocr_results, page_number).await?
        }
        SegmentationStrategy::Page => page_segmentation(page, ocr_results, page_number).await?,
    };
    Ok(segments)
}

/// Process the pages
///
/// This function will perform OCR, segmentation and chunking on the pages
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn std::error::Error>> {
    pipeline
        .update_status(Status::Processing, Some("Segmentation and OCR".to_string()))
        .await?;
    let pdf_ocr_results = match pdf::extract_ocr_results(pipeline.pdf_file.as_ref().unwrap()) {
        Ok(ocr_results) => ocr_results,
        Err(e) => {
            println!("Error getting pdf ocr results: {:?}", e);
            vec![vec![]; pipeline.page_count.unwrap_or(0) as usize]
        }
    };
    println!("pdf_ocr_results: {:?}", pdf_ocr_results);

    let page_segments = try_join_all(
        pipeline
            .page_images
            .as_ref()
            .unwrap()
            .par_iter()
            .enumerate()
            .map(|(page_idx, page)| {
                process_page(
                    page,
                    pipeline.task_payload.clone().unwrap(),
                    pdf_ocr_results[page_idx].clone(),
                    page_idx as u32 + 1,
                )
            })
            .collect::<Vec<_>>(),
    )
    .await?;

    pipeline
        .update_status(Status::Processing, Some("Chunking".to_string()))
        .await?;

    let chunks = chunking::hierarchical_chunking(
        page_segments.into_iter().flatten().collect(),
        pipeline
            .task_payload
            .as_ref()
            .unwrap()
            .current_configuration
            .chunk_processing
            .target_length,
    )?;

    pipeline.chunks = Some(chunks);
    Ok(())
}
