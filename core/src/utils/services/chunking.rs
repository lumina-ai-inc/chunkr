use crate::models::chunkr::output::{Chunk, Segment, SegmentType};

pub fn hierarchical_chunking(
    segments: Vec<Segment>,
    target_length: i32,
) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
    let mut chunks: Vec<Chunk> = Vec::new();
    if target_length == 0 || target_length == 1 {
        for segment in segments {
            chunks.push(Chunk::new(vec![segment.clone()]));
        }
    } else {
        let mut current_segments: Vec<Segment> = Vec::new();
        let mut current_word_count = 0;

        fn finalize_and_start_new_chunk(chunks: &mut Vec<Chunk>, segments: &mut Vec<Segment>) {
            if !segments.is_empty() {
                chunks.push(Chunk::new(segments.clone()));
                segments.clear();
            }
        }

        for segment in segments.iter() {
            let segment_word_count = segment.content.split_whitespace().count() as i32;
            match segment.segment_type {
                // titles and section headers must start a new chunk
                SegmentType::Title | SegmentType::SectionHeader => {
                    finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                    current_segments.push(segment.clone());
                    current_word_count = segment_word_count;
                    continue;
                }
                // headers and footers are 1 chunk each
                SegmentType::PageHeader | SegmentType::PageFooter => {
                    finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                    current_segments.push(segment.clone());
                    finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                    current_word_count = 0;
                    continue;
                }
                _ => {}
            }

            if current_word_count + segment_word_count > target_length {
                finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                current_segments.push(segment.clone());
                current_word_count = segment_word_count;
            } else {
                current_segments.push(segment.clone());
                current_word_count += segment_word_count;
            }
        }

        finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
    }

    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_process_bounding_boxes() -> Result<(), Box<dyn std::error::Error>> {
        async fn process_bounding_boxes(
            file_path: &str,
            target_size: usize,
        ) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
            let file_content = tokio::fs::read_to_string(file_path).await?;
            let segments: Vec<Segment> = serde_json::from_str(&file_content)?;
            Ok(hierarchical_chunking(segments, target_size as i32)?)
        }

        // Load the bounding_boxes.json file
        let mut input_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        input_path
            .push("/Users/ishaankapoor/chunk-my-docs/example/input/no_chunk_bounding_boxes.json");
        let input_file_path = input_path.to_str().unwrap();

        // Process the bounding boxes
        let chunks = process_bounding_boxes(input_file_path, 0).await?;

        // Save output to output/test.json in example folder
        let mut output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        output_path.push("/Users/ishaankapoor/chunk-my-docs/example/output/test.json");
        let output_file_path = output_path.to_str().unwrap();

        // Serialize chunks to JSON
        let json_output = serde_json::to_string_pretty(&chunks)?;

        // Write JSON to file
        fs::write(output_file_path, json_output)?;

        println!("Output saved to: {}", output_file_path);
        Ok(())
    }
}
