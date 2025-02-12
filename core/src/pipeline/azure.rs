use crate::models::chunkr::azure::DocumentAnalysisFeature;
use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::services::azure::perform_azure_analysis;

/// Use Azure document layout analysis to perform segmentation and ocr
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn std::error::Error>> {
    pipeline
        .get_task()?
        .update(
            Some(Status::Processing),
            Some("Running Azure analysis".to_string()),
            None,
            None,
            None,
            None,
            None,
        )
        .await?;

    let pdf_file = pipeline.pdf_file.as_ref().ok_or("PDF file not found")?;
    let configuration = pipeline.get_task()?.configuration.clone();
    let features = if configuration.high_resolution {
        Some(vec![DocumentAnalysisFeature::OcrHighResolution])
    } else {
        None
    };
    let chunks =
        perform_azure_analysis(&pdf_file, features, configuration.segmentation_strategy).await?;

    pipeline.chunks = chunks;
    Ok(())
}
