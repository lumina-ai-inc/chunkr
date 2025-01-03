use crate::models::chunkr::output::{OCRResult, Segment};
use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::{Status, TaskPayload};
use crate::models::chunkr::upload::OcrStrategy;
use crate::utils::services::chunking;
use crate::utils::services::ocr;
use crate::utils::services::pdf;
use crate::utils::services::segmentation;
use futures::future::try_join_all;
use std::sync::Arc;
use tempfile::NamedTempFile;

async fn ocr_page_all(
    page: Arc<NamedTempFile>,
) -> Result<Vec<OCRResult>, Box<dyn std::error::Error>> {
    match ocr::perform_general_ocr(page.clone()).await {
        Ok(ocr_results) => Ok(ocr_results),
        Err(e) => {
            println!("Error in performing OCR: {:?}", e);
            Err(e.to_string().into())
        }
    }
}

async fn ocr_page_auto(
    page: Arc<NamedTempFile>,
    extracted_ocr_result: Vec<OCRResult>,
) -> Result<Vec<OCRResult>, Box<dyn std::error::Error>> {
    if extracted_ocr_result.is_empty() {
        match ocr::perform_general_ocr(page.clone()).await {
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

async fn process_page(
    page: Arc<NamedTempFile>,
    task_payload: TaskPayload,
    extracted_ocr_result: Vec<OCRResult>,
    page_number: u32,
) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
    let ocr_result = match task_payload.current_configuration.ocr_strategy {
        OcrStrategy::All => ocr_page_all(Arc::clone(&page)).await?,
        OcrStrategy::Auto => ocr_page_auto(Arc::clone(&page), extracted_ocr_result).await?,
    };

    let segments = match segmentation::perform_segmentation(
        Arc::clone(&page),
        ocr_result,
        page_number,
    )
    .await
    {
        Ok(segments) => segments,
        Err(e) => {
            println!("Error in performing segmentation: {:?}", e);
            return Err(e.to_string().into());
        }
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

    let segments = try_join_all(pipeline.pages.as_ref().unwrap().iter().enumerate().map(
        |(page_idx, page)| {
            process_page(
                page.clone(),
                pipeline.task_payload.clone().unwrap(),
                pdf_ocr_results[page_idx].clone(),
                page_idx as u32 + 1,
            )
        },
    ))
    .await?;

    pipeline
        .update_status(Status::Processing, Some("Chunking".to_string()))
        .await?;

    let chunks = chunking::hierarchical_chunking(
        segments.into_iter().flatten().collect(),
        pipeline
            .task_payload
            .as_ref()
            .unwrap()
            .current_configuration
            .target_chunk_length,
    )?;

    pipeline.chunks = Some(chunks);
    Ok(())
}
