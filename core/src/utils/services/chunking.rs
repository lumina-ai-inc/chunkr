use crate::models::chunkr::output::{Chunk, Segment, SegmentType};

fn get_hierarchy_level(segment_type: &SegmentType) -> i32 {
    match segment_type {
        SegmentType::Title => 3,
        SegmentType::SectionHeader => 2,
        _ => 1,
    }
}

pub fn hierarchical_chunking(
    segments: Vec<Segment>,
    target_length: i32,
    ignore_headers_and_footers: bool,
) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut current_segments: Vec<Segment> = Vec::new();
    let mut current_word_count = 0;

    fn finalize_and_start_new_chunk(chunks: &mut Vec<Chunk>, segments: &mut Vec<Segment>) {
        if !segments.is_empty() {
            chunks.push(Chunk::new(segments.clone()));
            segments.clear();
        }
    }

    let mut prev_hierarchy_level = 1;
    let mut segment_paired = false;

    for (i, segment) in segments.iter().enumerate() {
        let segment_word_count = segment.content.split_whitespace().count() as i32;
        let current_hierarchy_level = get_hierarchy_level(&segment.segment_type);

        match segment.segment_type {
            SegmentType::Title | SegmentType::SectionHeader => {
                if current_hierarchy_level > prev_hierarchy_level {
                    finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                }
                current_segments.push(segment.clone());
                current_word_count = segment_word_count;
            }
            SegmentType::PageHeader | SegmentType::PageFooter => {
                if ignore_headers_and_footers {
                    continue;
                }
                finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                current_segments.push(segment.clone());
                finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                current_word_count = 0;
            }
            _ => {
                let mut default_chunk_behavior = true;

                if (segment.segment_type == SegmentType::Picture || segment.segment_type == SegmentType::Table) && !segment_paired {
                    let next_is_caption = segments.get(i + 1).map_or(false, |s| s.segment_type == SegmentType::Caption);
                    let caption_word_count = segments.get(i + 1).map_or(0, |s| s.content.split_whitespace().count() as i32);
                    if next_is_caption {
                        if current_word_count + segment_word_count + caption_word_count > target_length {
                            finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                            current_segments.push(segment.clone());
                            current_word_count = segment_word_count;
                            default_chunk_behavior = false;
                            segment_paired = true; // Caption is paired with picture or table
                        }
                    }
                }
                if segment.segment_type == SegmentType::Caption && !segment_paired {
                    let next_is_asset = segments.get(i + 1).map_or(false, |s| s.segment_type == SegmentType::Picture || s.segment_type == SegmentType::Table);
                    let asset_word_count = segments.get(i + 1).map_or(0, |s| s.content.split_whitespace().count() as i32);
                    if next_is_asset {
                        if current_word_count + segment_word_count + asset_word_count > target_length {
                            finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                            current_segments.push(segment.clone());
                            current_word_count = segment_word_count;
                            default_chunk_behavior = false;
                            segment_paired = true; // Picture or table is paired with caption
                        }
                    }
                }
                if default_chunk_behavior {
                    if current_word_count + segment_word_count > target_length {
                        finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                        current_segments.push(segment.clone());
                        current_word_count = segment_word_count;
                    } else {
                        current_segments.push(segment.clone());
                        current_word_count += segment_word_count;
                    }
                    segment_paired = false; // Reset segment pairing after default chunk behavior
                }
            }
        }

        prev_hierarchy_level = current_hierarchy_level;
    }

    finalize_and_start_new_chunk(&mut chunks, &mut current_segments);

    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::chunkr::output::{BoundingBox, Segment, SegmentType};

    fn create_segment(content: &str, segment_type: SegmentType) -> Segment {
        Segment {
            bbox: BoundingBox::new(0.0, 0.0, 0.0, 0.0),
            confidence: None,
            content: content.to_string(),
            html: String::new(),
            image: None,
            llm: None,
            markdown: String::new(),
            ocr: None,
            page_height: 0.0,
            page_width: 0.0,
            page_number: 0,
            segment_id: "".to_string(),
            segment_type,
        }
    }

    #[test]
    fn test_caption_picture_pairing() {
        let segments = vec![
            create_segment("Caption 1", SegmentType::Caption),
            create_segment("Picture 1", SegmentType::Picture),
            create_segment("Caption 2", SegmentType::Caption),
            create_segment("Picture 2", SegmentType::Picture),
        ];

        let chunks = hierarchical_chunking(segments, 100, true).unwrap();

        // Verify that captions stay with their pictures
        assert_eq!(chunks[0].segments.len(), 4);
        assert_eq!(chunks[0].segments[0].segment_type, SegmentType::Caption);
        assert_eq!(chunks[0].segments[1].segment_type, SegmentType::Picture);
        assert_eq!(chunks[0].segments[2].segment_type, SegmentType::Caption);
        assert_eq!(chunks[0].segments[3].segment_type, SegmentType::Picture);
    }

    #[test]
    fn test_picture_table_caption_precedence() {
        let segments = vec![
            create_segment("Table 1", SegmentType::Table),
            create_segment("Caption 1", SegmentType::Caption),
            create_segment("Picture 1", SegmentType::Picture),
        ];

        let chunks = hierarchical_chunking(segments, 100, true).unwrap();

        // Verify that caption pairs with picture, not table
        assert_eq!(chunks[0].segments.len(), 3);
        assert_eq!(chunks[0].segments[0].segment_type, SegmentType::Table);
        assert_eq!(chunks[0].segments[1].segment_type, SegmentType::Caption);
        assert_eq!(chunks[0].segments[2].segment_type, SegmentType::Picture);
    }

    #[test]
    fn test_chunk_boundary_respect() {
        let segments = vec![
            create_segment("Some long text ".repeat(10).as_str(), SegmentType::Text),
            create_segment("Caption 1", SegmentType::Caption),
            create_segment("Picture 1", SegmentType::Picture),
        ];

        let chunks = hierarchical_chunking(segments, 15, true).unwrap();

        // Verify that caption-picture pair stays together in new chunk
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].segments[0].segment_type, SegmentType::Text);
        assert_eq!(chunks[1].segments[0].segment_type, SegmentType::Caption);
        assert_eq!(chunks[1].segments[1].segment_type, SegmentType::Picture);
    }

    #[test]
    fn test_complex_sequence() {
        let segments = vec![
            create_segment("Caption 1", SegmentType::Caption),
            create_segment("Picture 1", SegmentType::Picture),
            create_segment("Picture 2", SegmentType::Picture),
            create_segment("Caption 2", SegmentType::Caption),
            create_segment("Some text", SegmentType::Text),
        ];

        let chunks = hierarchical_chunking(segments, 100, true).unwrap();

        // Verify correct pairing in complex sequence
        assert_eq!(chunks[0].segments.len(), 5);
        assert_eq!(chunks[0].segments[0].segment_type, SegmentType::Caption);
        assert_eq!(chunks[0].segments[1].segment_type, SegmentType::Picture);
        assert_eq!(chunks[0].segments[2].segment_type, SegmentType::Picture);
        assert_eq!(chunks[0].segments[3].segment_type, SegmentType::Caption);
        assert_eq!(chunks[0].segments[4].segment_type, SegmentType::Text);
    }

    #[test]
    fn test_unpaired_elements() {
        let segments = vec![
            create_segment("Picture 1", SegmentType::Picture),
            create_segment("Some text", SegmentType::Text),
            create_segment("Caption 1", SegmentType::Caption),
        ];

        let chunks = hierarchical_chunking(segments, 100, true).unwrap();

        // Verify unpaired elements are treated as regular segments
        assert_eq!(chunks[0].segments.len(), 3);
        assert_eq!(chunks[0].segments[0].segment_type, SegmentType::Picture);
        assert_eq!(chunks[0].segments[1].segment_type, SegmentType::Text);
        assert_eq!(chunks[0].segments[2].segment_type, SegmentType::Caption);
    }

    #[test]
    fn test_hierarchical_chunking() {
        let segments = vec![
            create_segment("Title 1", SegmentType::Title),
            create_segment("Section 1", SegmentType::SectionHeader),
            create_segment("Some text", SegmentType::Text),
            create_segment("Section 2", SegmentType::SectionHeader),
            create_segment("More text", SegmentType::Text),
            create_segment("Title 2", SegmentType::Title),
            create_segment("Section 3", SegmentType::SectionHeader),
        ];

        let chunks = hierarchical_chunking(segments, 100, true).unwrap();

        // Debug print
        for (i, chunk) in chunks.iter().enumerate() {
            println!("Chunk {}:", i);
            for segment in &chunk.segments {
                println!("  {:?}", segment.segment_type);
            }
        }

        assert_eq!(chunks.len(), 3); // Updated expectation

        // First chunk: Title 1 + Section 1 + text
        assert_eq!(chunks[0].segments.len(), 3);
        assert_eq!(chunks[0].segments[0].segment_type, SegmentType::Title);
        assert_eq!(
            chunks[0].segments[1].segment_type,
            SegmentType::SectionHeader
        );
        assert_eq!(chunks[0].segments[2].segment_type, SegmentType::Text);

        // Second chunk: Section 2 + text
        assert_eq!(chunks[1].segments.len(), 2);
        assert_eq!(
            chunks[1].segments[0].segment_type,
            SegmentType::SectionHeader
        );
        assert_eq!(chunks[1].segments[1].segment_type, SegmentType::Text);

        // Third chunk: Title 2 + Section 3
        assert_eq!(chunks[2].segments.len(), 2);
        assert_eq!(chunks[2].segments[0].segment_type, SegmentType::Title);
        assert_eq!(
            chunks[2].segments[1].segment_type,
            SegmentType::SectionHeader
        );
    }

    #[test]
    fn test_caption_not_repeated() {
        let segments = vec![
            create_segment("Some regular text before.", SegmentType::Text),
            create_segment("Figure 1: A test caption", SegmentType::Caption),
            create_segment("[An image content placeholder]", SegmentType::Picture),
            create_segment("Some regular text after.", SegmentType::Text),
        ];

        let chunks = hierarchical_chunking(segments, 10, true).unwrap();
        
        println!("\n=== Chunk Contents ===");
        for (i, chunk) in chunks.iter().enumerate() {
            println!("\nChunk {}:", i);
            println!("{:?}", chunk.chunk_length);
            for segment in &chunk.segments {
                println!("  Type: {:?}, Content: \"{}\"", segment.segment_type, segment.content);
            }
        }

        let caption_count = chunks.iter()
            .filter(|chunk| chunk.segments.iter().any(|s| s.segment_type == SegmentType::Caption))
            .count();

        println!("\nTotal chunks with captions: {}", caption_count);

        assert_eq!(caption_count, 1, "Caption should appear exactly once in the chunks");
    }
}
