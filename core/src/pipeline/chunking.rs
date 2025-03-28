use crate::models::pipeline::Pipeline;
use crate::models::task::Status;
use crate::utils::services::chunking;
use rayon::prelude::*;

/// Chunk the segments
///
/// This function will perform chunking on the segments
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn std::error::Error>> {
    let mut task = pipeline.get_task()?;
    task.update(
        Some(Status::Processing),
        Some("Chunking".to_string()),
        None,
        None,
        None,
        None,
        None,
    )
    .await?;

    let mut chunks = pipeline.chunks.clone();

    if task.configuration.chunk_processing.target_length > 0 {
        let segments = chunks
            .clone()
            .into_iter()
            .flat_map(|c| c.segments)
            .collect();
        chunks = chunking::hierarchical_chunking(segments, &task.configuration)?;
    };

    chunks.par_iter_mut().for_each(|chunk| {
        chunk.generate_embed_text(&task.configuration);
    });

    pipeline.chunks = chunks;

    Ok(())
}
