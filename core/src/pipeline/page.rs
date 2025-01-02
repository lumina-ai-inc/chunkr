use crate::models::chunkr::output::OCRResult;
use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::models::chunkr::upload::OcrStrategy;
use crate::utils::services::ocr;
use crate::utils::services::pdf;
use futures::future::try_join_all;
use std::io::Error;

async fn perform_ocr_all(
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

async fn perform_ocr_auto(
    pipeline: &mut Pipeline,
) -> Result<Vec<Vec<OCRResult>>, Box<dyn std::error::Error>> {
    let pages = pipeline.pages.as_ref().unwrap();
    let pdf_ocr_results = pdf::extract_ocr_results(pipeline.pdf_file.as_ref())?;
    let pdf_ocr_results = &pdf_ocr_results;

    let futures = pages.iter().enumerate().map(|(index, page)| async move {
        if pdf_ocr_results
            .get(index)
            .map_or(true, |results| results.is_empty())
        {
            match ocr::perform_general_ocr(page.clone()).await {
                Ok(ocr_results) => Ok(ocr_results),
                Err(e) => Err(e.to_string()),
            }
        } else {
            Ok(pdf_ocr_results[index].clone())
        }
    });

    let ocr_results = try_join_all(futures).await?;
    Ok(ocr_results)
}

/// Process the pages
///
/// This function will perform OCR, segmentation and chunking on the pages
pub async fn process(
    pipeline: &mut Pipeline,
) -> Result<(Status, Option<String>), Box<dyn std::error::Error>> {
    // TODO: Implement OCR, segmentation and chunking
    let ocr_results = match pipeline.task_payload.current_configuration.ocr_strategy {
        OcrStrategy::All => perform_ocr_all(pipeline).await?,
        OcrStrategy::Auto => perform_ocr_auto(pipeline).await?,
    };
    Ok((Status::Processing, Some("Pages processed".to_string())))
}
