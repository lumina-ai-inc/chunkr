use crate::models::chunkr::output::{OCRResult, Segment};
use crate::models::chunkr::segmentation::ObjectDetectionResponse;
use std::error::Error;
use std::sync::Arc;
use tempfile::NamedTempFile;

pub async fn perform_segmentation(
    temp_file: Arc<NamedTempFile>,
    ocr_results: Vec<OCRResult>,
    page_number: u32,
) -> Result<Vec<Segment>, Box<dyn Error>> {
    let response: ObjectDetectionResponse = todo!();
    let segments = response.instances.to_segments(page_number, ocr_results);
    Ok(segments)
}
