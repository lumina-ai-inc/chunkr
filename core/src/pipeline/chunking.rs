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
        let mut segments: Vec<_> = chunks
            .clone()
            .into_iter()
            .flat_map(|c| c.segments)
            .collect();

        // Set embed fields for all segments before chunking
        segments.par_iter_mut().for_each(|segment| {
            if let Err(e) = segment.set_embed_field(&configuration) {
                println!("Error setting embed field: {e}");
            }
        });

        chunks = chunking::hierarchical_chunking(segments, &configuration, is_spreadsheet)?;
    } else {
        // If not doing hierarchical chunking, still set embed fields for existing segments
        chunks.par_iter_mut().for_each(|chunk| {
            chunk.segments.par_iter_mut().for_each(|segment| {
                if let Err(e) = segment.set_embed_field(&configuration) {
                    println!("Error setting embed field: {e}");
                }
            });
        });
    }

    // Generate chunk-level embed text from already-set segment embed fields
    chunks.par_iter_mut().for_each(|chunk| {
        chunk.generate_embed_text(&configuration);
    });

    pipeline.chunks = chunks;

    Ok(())
}
