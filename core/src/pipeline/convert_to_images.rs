use crate::configs::worker_config::Config as WorkerConfig;
use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::services::pdf::pages_as_images;
use std::sync::Arc;

/// Convert the PDF to images
///
/// This function will convert the PDF to images and store the images in the pipeline
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn std::error::Error>> {
    let worker_config = WorkerConfig::from_env()?;
    pipeline
        .update_remote_status(Status::Processing, Some("Converting to images".to_string()))
        .await?;
    let task_payload = pipeline.task_payload.as_mut().unwrap();
    let scaling_factor = match task_payload.current_configuration.high_resolution {
        true => worker_config.high_res_scaling_factor,
        false => 1.0,
    };
    let pages = pages_as_images(pipeline.pdf_file.as_ref().unwrap(), scaling_factor)?;
    pipeline.page_images = Some(pages.into_iter().map(|p| Arc::new(p)).collect());
    Ok(())
}
