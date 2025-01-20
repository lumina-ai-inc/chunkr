use crate::models::chunkr::output::Segment;
use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::services::chunking;

/// Use Azure document layout analysis to perform segmentation and ocr
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn std::error::Error>> {
    pipeline
        .get_task()?
        .update(
            Some(Status::Processing),
            Some("Chunking".to_string()),
            None,
            None,
            None,
            None,
        )
        .await?;

    let segments: Vec<Segment> = pipeline
        .output
        .chunks
        .clone()
        .into_iter()
        .map(|c| c.segments)
        .flatten()
        .collect();

    let chunk_processing = pipeline.get_task()?.configuration.chunk_processing.clone();

    let chunks = chunking::hierarchical_chunking(
        segments,
        chunk_processing.target_length,
        chunk_processing.ignore_headers_and_footers,
    )?;

    pipeline.output.chunks = chunks;
    Ok(())
}
