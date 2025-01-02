use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::services::pdf::pages_as_images;
use std::sync::Arc;

/// Convert the PDF to images
///
/// This function will convert the PDF to images and store the images in the pipeline
pub async fn process(
    pipeline: &mut Pipeline,
) -> Result<(Status, Option<String>), Box<dyn std::error::Error>> {
    let pages = pages_as_images(pipeline.pdf_file.as_ref())?;
    pipeline.pages = Some(pages.into_iter().map(|p| Arc::new(p)).collect());
    Ok((Status::Processing, Some("Converted to images".to_string())))
}
