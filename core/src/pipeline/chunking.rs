use crate::models::pipeline::Pipeline;
use crate::utils::services::chunking;
use rayon::prelude::*;

/// Chunk the segments
///
/// This function will perform chunking on the segments
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn std::error::Error>> {
    let configuration = pipeline.get_task()?.configuration.clone();
    let mut chunks = pipeline.chunks.clone();
    let is_spreadsheet = pipeline.get_task()?.is_spreadsheet;
    println!("Tokenizer: {:?}", configuration.chunk_processing.tokenizer);
    if configuration.chunk_processing.target_length > 0 {
        let segments = chunks
            .clone()
            .into_iter()
            .flat_map(|c| c.segments)
            .collect();
        chunks = chunking::hierarchical_chunking(segments, &configuration, is_spreadsheet)?;
    };

    chunks.par_iter_mut().for_each(|chunk| {
        chunk.generate_embed_text(&configuration);
    });

    pipeline.chunks = chunks;

    Ok(())
}
