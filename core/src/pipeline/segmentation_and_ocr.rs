use crate::models::chunkr::output::{BoundingBox, OCRResult, Segment, SegmentType};
use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::{Configuration, Status};
use crate::models::chunkr::upload::{OcrStrategy, SegmentationStrategy};
use crate::utils::services::chunking;
use crate::utils::services::images;
use crate::utils::services::ocr;
use crate::utils::services::pdf;
use crate::utils::services::segmentation;
use futures::future::try_join_all;
use rayon::prelude::*;
use tempfile::NamedTempFile;

async fn ocr_page_all(
    page: &NamedTempFile,
) -> Result<Vec<OCRResult>, Box<dyn std::error::Error + Send + Sync>> {
    Ok(ocr::perform_general_ocr(&page).await?)
}

async fn ocr_page_auto(
    page: &NamedTempFile,
    extracted_ocr_result: Vec<OCRResult>,
) -> Result<Vec<OCRResult>, Box<dyn std::error::Error + Send + Sync>> {
    if extracted_ocr_result.is_empty() {
        Ok(ocr::perform_general_ocr(&page).await?)
    } else {
        Ok(extracted_ocr_result)
    }
}

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

async fn process_page(
    page: &NamedTempFile,
    configuration: Configuration,
    extracted_ocr_result: Vec<OCRResult>,
    page_number: u32,
) -> Result<Vec<Segment>, Box<dyn std::error::Error + Send + Sync>> {
    let ocr_results: Vec<OCRResult> = match configuration.ocr_strategy {
        OcrStrategy::All => ocr_page_all(&page).await,
        OcrStrategy::Auto => ocr_page_auto(&page, extracted_ocr_result).await,
    }?;

    let segments = match configuration.segmentation_strategy {
        SegmentationStrategy::LayoutAnalysis => {
            segmentation::perform_segmentation(page, ocr_results, page_number).await
        }
        SegmentationStrategy::Page => page_segmentation(page, ocr_results, page_number).await,
    }?;

    Ok(segments)
}

/// Process the pages
///
/// This function will perform OCR, segmentation and chunking on the pages
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn std::error::Error>> {
    pipeline
        .get_task()?
        .update(
            Some(Status::Processing),
            Some("Segmentation and OCR".to_string()),
            None,
            None,
            None,
            None,
        )
        .await?;
    let pdf_ocr_results = match pdf::extract_ocr_results(pipeline.pdf_file.as_ref().unwrap()) {
        Ok(ocr_results) => ocr_results,
        Err(e) => {
            println!("Error getting pdf ocr results: {:?}", e);
            vec![vec![]; pipeline.get_task()?.page_count.unwrap_or(0) as usize]
        }
    };

    let page_segments = match try_join_all(
        pipeline
            .page_images
            .as_ref()
            .unwrap()
            .par_iter()
            .enumerate()
            .map(|(page_idx, page)| {
                let task = pipeline.get_task().unwrap();
                process_page(
                    page,
                    task.configuration.clone(),
                    pdf_ocr_results[page_idx].clone(),
                    page_idx as u32 + 1,
                )
            })
            .collect::<Vec<_>>(),
    )
    .await
    {
        Ok(page_segments) => page_segments,
        Err(e) => {
            println!("Error in performing segmentation and OCR: {:?}", e);
            return Err(e.to_string().into());
        }
    };

    pipeline
        .get_task()?
        .update(
            Some(Status::Processing),
            Some("Chunking".to_string()),
            None,
            None,
            None,
            None,
        )
        .await?;

    // If segmentation is performed differently, we need to update the chunking step
    // TODO: Move to a separate step - need to figure out how to add segmentation results to pipeline to be able to use it in the chunking step.
    // add segmentation output as if chunking is off and then:
    // 1.  the orchestrator optionally adds the segmentation step and optionally adds the chunking step (theoretically faster)
    //      - Requires diff check of segmentation and chunking
    // 2. the orchestrator optionally adds the segmentation step and we can always chunk (is very fast so prob doesn't matter)
    //      - Only diff check of segmentation
    let chunk_processing = pipeline.get_task()?.configuration.chunk_processing.clone();

    let chunks = chunking::hierarchical_chunking(
        page_segments.into_iter().flatten().collect(),
        chunk_processing.target_length,
        chunk_processing.ignore_headers_and_footers,
    )?;

    pipeline.output.chunks = chunks;
    Ok(())
}
