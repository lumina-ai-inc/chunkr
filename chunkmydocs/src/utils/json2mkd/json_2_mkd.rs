use crate::models::server::segment::{Chunk, Segment, SegmentType};
use std::collections::VecDeque;

pub async fn hierarchical_chunk_and_add_markdown(
    segments: Vec<Segment>,
    target_length: Option<i32>,
) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut current_chunk: Vec<Segment> = Vec::new();
    let mut current_word_count = 0;
    let mut hierarchy_stack: VecDeque<SegmentType> = VecDeque::new();
    let target_length = target_length.unwrap_or(512);

    fn finalize_chunk(chunk: &mut Vec<Segment>, chunks: &mut Vec<Chunk>) {
        if !chunk.is_empty() {
            chunks.push(Chunk {
                segments: chunk.clone(),
                markdown: generate_markdown(chunk),
            });
            chunk.clear();
        }
    }

    for (index, segment) in segments.iter().enumerate() {
        let segment_word_count = segment.text.split_whitespace().count() as i32;

        match segment.segment_type {
            SegmentType::Title => {
                finalize_chunk(&mut current_chunk, &mut chunks);
                current_word_count = 0;
                hierarchy_stack.clear();
                hierarchy_stack.push_front(segment.segment_type.clone());
            }
            SegmentType::SectionHeader => {
                if current_word_count + segment_word_count > target_length {
                    finalize_chunk(&mut current_chunk, &mut chunks);
                    current_word_count = 0;
                }
                while let Some(top) = hierarchy_stack.front() {
                    if *top == SegmentType::Title {
                        break;
                    }
                    hierarchy_stack.pop_front();
                }
                hierarchy_stack.push_front(segment.segment_type.clone());
            }
            SegmentType::PageHeader | SegmentType::PageFooter => {
                finalize_chunk(&mut current_chunk, &mut chunks);
                chunks.push(Chunk {
                    segments: vec![segment.clone()],
                    markdown: generate_markdown(&[segment.clone()]),
                });
                continue;
            }
            _ => {
                if current_word_count + segment_word_count > target_length {
                    if current_chunk.last().map(|s| s.segment_type.clone())
                        == Some(SegmentType::SectionHeader)
                    {
                        // If the last segment is a section header, keep it with this text
                        let header = current_chunk.pop().unwrap();
                        finalize_chunk(&mut current_chunk, &mut chunks);
                        current_word_count = header.text.split_whitespace().count() as i32;
                        current_chunk.push(header);
                    } else {
                        finalize_chunk(&mut current_chunk, &mut chunks);
                        current_word_count = 0;
                    }
                }
            }
        }

        current_chunk.push(segment.clone());
        current_word_count += segment_word_count;

        // Look ahead for section header + text pairs
        if segment.segment_type == SegmentType::SectionHeader && index + 1 < segments.len() {
            if let SegmentType::Text = segments[index + 1].segment_type {
                let next_word_count = segments[index + 1].text.split_whitespace().count() as i32;
                if current_word_count + next_word_count > target_length {
                    finalize_chunk(&mut current_chunk, &mut chunks);
                    current_word_count = 0;
                }
            }
        }
    }

    // Add the last chunk if it's not empty
    finalize_chunk(&mut current_chunk, &mut chunks);

    // Remove any empty chunks (shouldn't be necessary, but just in case)
    chunks.retain(|chunk| !chunk.segments.is_empty());

    Ok(chunks)
}

fn generate_markdown(segments: &[Segment]) -> String {
    let mut markdown = String::new();
    let mut list_level = 0;

    for segment in segments {
        let segment_type = &segment.segment_type;

        match segment_type {
            SegmentType::Title => {
                list_level = 0;
                markdown.push_str(&format!("# {}\n\n", segment.text.trim()));
            }
            SegmentType::SectionHeader => {
                list_level = 0;
                markdown.push_str(&format!("## {}\n\n", segment.text.trim()));
            }
            SegmentType::Text => {
                list_level = 0;
                markdown.push_str(&format!("{}\n\n", segment.text.trim()));
            }
            SegmentType::ListItem => {
                if list_level == 0 {
                    list_level = 1;
                }
                markdown.push_str(&format!(
                    "{}- {}\n",
                    "  ".repeat(list_level - 1),
                    segment.text.trim()
                ));
            }
            SegmentType::Caption => {
                list_level = 0;
                markdown.push_str(&format!("*{}*\n\n", segment.text.trim()));
            }
            SegmentType::Table => {
                list_level = 0;
                markdown.push_str(&format!("```\n{}\n```\n\n", segment.text.trim()));
            }
            SegmentType::Formula => {
                list_level = 0;
                markdown.push_str(&format!("${}$\n\n", segment.text.trim()));
            }
            SegmentType::Picture => {
                list_level = 0;
                markdown.push_str(&format!("![Image]({})\n\n", segment.text.trim()));
            }
            SegmentType::PageHeader => {
                markdown.push_str(&format!("*Page Header:* {}\n\n", segment.text.trim()));
            }
            SegmentType::PageFooter => {
                markdown.push_str(&format!("*Page Footer:* {}\n\n", segment.text.trim()));
            }
            SegmentType::Footnote => {
                list_level = 0;
                markdown.push_str(&format!("[^1]: {}\n\n", segment.text.trim()));
            }
        }
    }

    markdown.trim().to_string()
}

pub async fn process_bounding_boxes(
    file_path: &str,
    target_size: usize,
) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
    let file_content = tokio::fs::read_to_string(file_path).await?;
    let segments: Vec<Segment> = serde_json::from_str(&file_content)?;
    hierarchical_chunk_and_add_markdown(segments, Some(target_size as i32)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_process_bounding_boxes() -> Result<(), Box<dyn std::error::Error>> {
        // Load the bounding_boxes.json file
        let mut input_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        input_path
            .push("/Users/ishaankapoor/chunk-my-docs/example/input/no_chunk_bounding_boxes.json");
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
