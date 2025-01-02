use std::io::Error;

use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::services::ocr;
use futures::future::try_join_all;

/// Process the pages
///
/// This function will perform OCR, segmentation and chunking on the pages
pub async fn process(
    pipeline: &mut Pipeline,
) -> Result<(Status, Option<String>), Box<dyn std::error::Error>> {
    // TODO: Implement OCR, segmentation and chunking
    try_join_all(
        pipeline
            .pages
            .as_ref()
            .unwrap_or(&vec![])
            .iter()
            .map(|page| ocr::perform_general_ocr(page.clone())),
    )
    .await
    .map_err(|e| {
        println!("Error in performing OCR: {:?}", e);
        Error::new(std::io::ErrorKind::Other, "Error in performing OCR")
    })?;

    Ok((Status::Processing, Some("Pages processed".to_string())))
}
