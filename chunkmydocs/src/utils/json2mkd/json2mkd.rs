use crate::models::extraction::segment::{Chunk, Segment, SegmentType};

pub async fn chunk_and_add_markdown(
    segments: Vec<Segment>,
    target_length: usize,
) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut current_chunk: Vec<Segment> = Vec::new();
    let mut current_word_count = 0;

    for segment in segments {
        let segment_word_count = segment.text.split_whitespace().count();

        if current_word_count + segment_word_count > target_length && !current_chunk.is_empty() {
            chunks.push(Chunk {
                segments: current_chunk.clone(),
                markdown: generate_markdown(&current_chunk),
            });
            current_chunk.clear();
            current_word_count = 0;
        }

        current_chunk.push(segment);
        current_word_count += segment_word_count;
    }

    // Add the last chunk if it's not empty
    if !current_chunk.is_empty() {
        chunks.push(Chunk {
            segments: current_chunk.clone(),
            markdown: generate_markdown(&current_chunk),
        });
    }

    Ok(chunks)
}

fn generate_markdown(segments: &[Segment]) -> String {
    let mut markdown = String::new();
    println!("test");
    for segment in segments {
        let segment_type = &segment.segment_type;

        match segment_type {
            SegmentType::Title => markdown.push_str(&format!("# {}\n\n", segment.text)),
            SegmentType::SectionHeader => markdown.push_str(&format!("## {}\n\n", segment.text)),
            SegmentType::Text => markdown.push_str(&format!("{}\n\n", segment.text)),
            SegmentType::ListItem => markdown.push_str(&format!("- {}\n", segment.text)),
            SegmentType::Caption => markdown.push_str(&format!("*{}\n\n", segment.text)),
            SegmentType::Table => markdown.push_str(&format!("```\n{}\n```\n\n", segment.text)),
            SegmentType::Formula => markdown.push_str(&format!("${}$\n\n", segment.text)),
            SegmentType::Picture => markdown.push_str(&format!("![Image]({})\n\n", segment.text)),
            SegmentType::PageHeader | SegmentType::PageFooter => {} // Ignore these types
            SegmentType::Footnote => markdown.push_str(&format!("[^1]: {}\n\n", segment.text)),
        }
    }

    markdown.trim().to_string()
}
pub async fn process_bounding_boxes(
    file_path: &str,
    target_size: usize,
) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
    println!("Processing file: {}", file_path);
    let file_content = tokio::fs::read_to_string(file_path).await?;
    println!("File content loaded, length: {}", file_content.len());

    let mut segments: Vec<Segment> = serde_json::from_str(&file_content)?;
    println!("Parsed {} segments", segments.len());
    println!("Segment types processed");
    chunk_and_add_markdown(segments, target_size).await
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::fs;

    #[tokio::test]
    async fn test_process_bounding_boxes() -> Result<(), Box<dyn std::error::Error>> {
        // Load the bounding_boxes.json file
        let mut input_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        input_path.push(
            "/Users/ishaankapoor/chunk-my-docs/example/input/no_chunk_bounding_boxes.json",
        );
        let input_file_path = input_path.to_str().unwrap();

        // Process the bounding boxes
        let chunks = process_bounding_boxes(input_file_path, 512).await?;

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