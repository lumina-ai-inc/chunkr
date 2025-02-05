use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::services::pdf::pages_as_images;
use std::sync::Arc;

/// Convert the PDF to images
///
/// This function will convert the PDF to images and store the images in the pipeline
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn std::error::Error>> {
    pipeline
        .get_task()?
        .update(
            Some(Status::Processing),
            Some("Converting to images".to_string()),
            None,
            None,
            None,
            None,
            None,
        )
        .await?;
    let scaling_factor = pipeline.get_task()?.configuration.get_scaling_factor()?;
    let pages = pages_as_images(pipeline.pdf_file.as_ref().unwrap(), scaling_factor)?;
    pipeline.page_images = Some(pages.into_iter().map(|p| Arc::new(p)).collect());
    Ok(())
}
