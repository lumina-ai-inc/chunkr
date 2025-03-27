use crate::models::azure::DocumentAnalysisFeature;
use crate::models::pipeline::Pipeline;
use crate::models::task::Status;
use crate::utils::services::azure::perform_azure_analysis;
use rayon::prelude::*;

/// Use Azure document layout analysis to perform segmentation and ocr
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn std::error::Error>> {
    let mut task = pipeline.get_task()?;
    task.update(
        Some(Status::Processing),
        Some("Running Azure analysis".to_string()),
        None,
        None,
        None,
        None,
        None,
    )
    .await?;

    let configuration = pipeline.get_task()?.configuration.clone();
    let scaling_factor = pipeline.get_scaling_factor()?;
    let features = if configuration.high_resolution {
        Some(vec![DocumentAnalysisFeature::OcrHighResolution])
    } else {
        None
    };

    let file = pipeline.get_file()?;
    let mut chunks =
        perform_azure_analysis(&file, features, configuration.segmentation_strategy).await?;
    chunks.par_iter_mut().for_each(|chunk| {
        chunk.segments.par_iter_mut().for_each(|segment| {
            segment.scale(scaling_factor);
        });
    });
    pipeline.chunks = chunks;
    Ok(())
}
