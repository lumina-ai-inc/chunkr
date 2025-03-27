use crate::models::pipeline::Pipeline;
use crate::models::task::Status;
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

    let file = pipeline.get_file()?;

    if pipeline.get_mime_type()?.starts_with("image/") {
        pipeline.page_images = Some(vec![file.clone()]);
    } else {
        let scaling_factor = pipeline.get_scaling_factor()?;
        let pages = pages_as_images(&file, scaling_factor)?;
        pipeline.page_images = Some(pages.into_iter().map(Arc::new).collect());
    }

    Ok(())
}
