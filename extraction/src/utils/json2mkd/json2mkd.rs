use crate::models::extraction::segment::{Chunk, Segment, SegmentType};
use crate::models::extraction::task::Status;

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
        let segment_type = match segment.segment_type.to_string().as_str() {
            "Title" => SegmentType::Title,
            "Section header" => SegmentType::SectionHeader,
            "Text" => SegmentType::Text,
            "List item" => SegmentType::ListItem,
            "Caption" => SegmentType::Caption,
            "Table" => SegmentType::Table,
            "Formula" => SegmentType::Formula,
            "Picture" => SegmentType::Picture,
            "Page header" => SegmentType::PageHeader,
            "Page footer" => SegmentType::PageFooter,
            _ => SegmentType::Text, // Default to Text for unknown types
        };

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
