use crate::models::chunkr::output::{OCRResult, OutputResponse, Segment};
use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::models::chunkr::upload::OcrStrategy;
use crate::utils::services::chunking;
use crate::utils::services::ocr;
use crate::utils::services::pdf;
use crate::utils::services::segmentation;
use futures::future::try_join_all;
use std::io::Error;
use tap::Pipe;

async fn ocr_pages_all(
    pipeline: &mut Pipeline,
) -> Result<Vec<Vec<OCRResult>>, Box<dyn std::error::Error>> {
    let ocr_results = try_join_all(
        pipeline
            .pages
            .as_ref()
            .unwrap()
            .iter()
            .map(|page| ocr::perform_general_ocr(page.clone())),
    )
    .await
    .map_err(|e| {
        println!("Error in performing OCR: {:?}", e);
        Error::new(std::io::ErrorKind::Other, "Error in performing OCR")
    })?;
    Ok(ocr_results)
}

async fn ocr_pages_auto(
    pipeline: &mut Pipeline,
) -> Result<Vec<Vec<OCRResult>>, Box<dyn std::error::Error>> {
    let pages = pipeline.pages.as_ref().unwrap();
    let pdf_ocr_results = match pdf::extract_ocr_results(pipeline.pdf_file.as_ref()) {
        Ok(ocr_results) => ocr_results,
        Err(e) => {
            println!("Error getting pdf ocr results: {:?}", e);
            vec![vec![]; pipeline.page_count.unwrap_or(0) as usize]
        }
    };
    let pdf_ocr_results = &pdf_ocr_results;

    let futures = pages.iter().enumerate().map(|(index, page)| async move {
        if pdf_ocr_results
            .get(index)
            .map_or(true, |results| results.is_empty())
        {
            match ocr::perform_general_ocr(page.clone()).await {
                Ok(ocr_results) => Ok(ocr_results),
                Err(e) => {
                    println!("Error in performing OCR: {:?}", e);
                    Err(e.to_string())
                }
            }
        } else {
            Ok(pdf_ocr_results[index].clone())
        }
    });

    let ocr_results = try_join_all(futures).await?;
    Ok(ocr_results)
}

async fn segment_pages(
    pipeline: &mut Pipeline,
) -> Result<Vec<Vec<Segment>>, Box<dyn std::error::Error>> {
    let futures = pipeline
        .pages
        .as_ref()
        .unwrap()
        .iter()
        .map(|page| segmentation::perform_segmentation(page.clone()));

    let segments = try_join_all(futures).await?;
    Ok(segments)
}

fn merge_segments_with_ocr(
    segments: Vec<Vec<Segment>>,
    ocr_results: Vec<Vec<OCRResult>>,
) -> Result<Vec<Vec<Segment>>, Box<dyn std::error::Error>> {
    segments
        .iter()
        .enumerate()
        .map(|(page_idx, page_segments)| {
            page_segments
                .iter()
                .map(|segment| {
                    let mut segment = segment.clone();
                    let empty_vec = vec![];
                    let page_ocr = ocr_results.get(page_idx).unwrap_or(&empty_vec);
                    let segment_ocr: Vec<OCRResult> = page_ocr
                        .iter()
                        .filter(|ocr| ocr.bbox.intersects(&segment.bbox))
                        .map(|ocr| {
                            let mut ocr = ocr.clone();
                            ocr.bbox.left -= segment.bbox.left;
                            ocr.bbox.top -= segment.bbox.top;
                            ocr
                        })
                        .collect();

                    segment.ocr = Some(segment_ocr);
                    segment
                })
                .collect()
        })
        .collect::<Vec<Vec<Segment>>>()
        .pipe(Ok)
}

/// Process the pages
///
/// This function will perform OCR, segmentation and chunking on the pages
pub async fn process(
    pipeline: &mut Pipeline,
) -> Result<(Status, Option<String>), Box<dyn std::error::Error>> {
    let ocr_results = match pipeline.task_payload.current_configuration.ocr_strategy {
        OcrStrategy::All => ocr_pages_all(pipeline).await?,
        OcrStrategy::Auto => ocr_pages_auto(pipeline).await?,
    };
    let segments = segment_pages(pipeline).await?;
    let segments_with_ocr = merge_segments_with_ocr(segments, ocr_results)?;
    let chunks = chunking::hierarchical_chunking(
        segments_with_ocr.into_iter().flatten().collect(),
        pipeline
            .task_payload
            .current_configuration
            .target_chunk_length,
    )?;
    pipeline.output = Some(OutputResponse {
        chunks,
        extracted_json: None,
    });
    Ok((Status::Processing, Some("Pages processed".to_string())))
}
