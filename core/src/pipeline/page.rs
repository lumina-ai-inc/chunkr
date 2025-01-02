use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;

/// Process the pages
/// 
/// This function will perform OCR, segmentation and chunking on the pages
pub async fn process(
    pipeline: &mut Pipeline,
) -> Result<(Status, Option<String>), Box<dyn std::error::Error>> {
    // TODO: Implement OCR, segmentation and chunking
    Ok((Status::Processing, Some("Pages processed".to_string())))
}
