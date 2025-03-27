use crate::models::chunkr::output::Segment;
use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::services::chunking;

/// Chunk the segments
///
/// This function will perform chunking on the segments
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
            None,
        )
        .await?;

    let segments: Vec<Segment> = pipeline
        .chunks
        .clone()
        .into_iter()
        .flat_map(|c| c.segments)
        .collect();

    let chunk_processing = pipeline.get_task()?.configuration.chunk_processing.clone();

    let chunks = chunking::hierarchical_chunking(
        segments,
        chunk_processing.target_length,
        chunk_processing.ignore_headers_and_footers,
    )?;

    pipeline.chunks = chunks;
    Ok(())
}
