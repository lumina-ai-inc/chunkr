use crate::models::chunkr::output::{Chunk, Segment, SegmentType};

fn find_caption_pair(
    segments: &[Segment],
    current_idx: usize,
    looking_forward: bool,
) -> Option<(usize, &Segment)> {
    let search_range = if looking_forward {
        segments.get((current_idx + 1)..)?
    } else {
        segments.get(..current_idx)?
    };

    for (offset, seg) in search_range.iter().enumerate() {
        // Stop searching if we hit a non-caption/picture/table segment
        if !matches!(
            seg.segment_type,
            SegmentType::Caption | SegmentType::Picture | SegmentType::Table
        ) {
            break;
        }

        if seg.segment_type == SegmentType::Caption {
            return Some((
                if looking_forward {
                    current_idx + 1 + offset
                } else {
                    current_idx - offset - 1
                },
                seg,
            ));
        }
    }
    None
}

fn should_pair_with_caption(
    segments: &[Segment],
    current_idx: usize,
    segment: &Segment,
) -> Option<(usize, i32)> {
    if segment.segment_type != SegmentType::Picture && segment.segment_type != SegmentType::Table {
        return None;
    }

    // Look for caption before and after
    let caption_before = find_caption_pair(segments, current_idx, false);
    let caption_after = find_caption_pair(segments, current_idx, true);

    // Determine which caption to pair with
    match (caption_before, caption_after) {
        (Some((before_idx, before_seg)), Some((after_idx, after_seg))) => {
            // If we're a picture, we take precedence over tables
            let before_has_table = segments[before_idx..current_idx]
                .iter()
                .any(|s| s.segment_type == SegmentType::Table);
            let after_has_table = segments[current_idx + 1..=after_idx]
                .iter()
                .any(|s| s.segment_type == SegmentType::Table);

            if segment.segment_type == SegmentType::Picture {
                // Picture takes the closest caption if there's no interference
                let before_distance = current_idx - before_idx;
                let after_distance = after_idx - current_idx;
                if before_distance <= after_distance && !before_has_table {
                    Some((
                        before_idx,
                        before_seg.content.split_whitespace().count() as i32,
                    ))
                } else if !after_has_table {
                    Some((
                        after_idx,
                        after_seg.content.split_whitespace().count() as i32,
                    ))
                } else {
                    None
                }
            } else {
                // Table only gets caption if there's no picture claiming it
                if !before_has_table && !after_has_table {
                    let before_distance = current_idx - before_idx;
                    let after_distance = after_idx - current_idx;
                    if before_distance <= after_distance {
                        Some((
                            before_idx,
                            before_seg.content.split_whitespace().count() as i32,
                        ))
                    } else {
                        Some((
                            after_idx,
                            after_seg.content.split_whitespace().count() as i32,
                        ))
                    }
                } else {
                    None
                }
            }
        }
        (Some((idx, seg)), None) => Some((idx, seg.content.split_whitespace().count() as i32)),
        (None, Some((idx, seg))) => Some((idx, seg.content.split_whitespace().count() as i32)),
        (None, None) => None,
    }
}

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

    let mut skip_indices = std::collections::HashSet::new();

    let mut prev_hierarchy_level = 1;

    for (i, segment) in segments.iter().enumerate() {
        if skip_indices.contains(&i) {
            continue;
        }

        let segment_word_count = segment.content.split_whitespace().count() as i32;
        let current_hierarchy_level = get_hierarchy_level(&segment.segment_type);

        match segment.segment_type {
            SegmentType::Title | SegmentType::SectionHeader => {
                if current_hierarchy_level > prev_hierarchy_level {
                    finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                }
                current_segments.push(segment.clone());
                current_word_count = segment_word_count;
                prev_hierarchy_level = current_hierarchy_level;
                continue;
            }
            SegmentType::PageHeader | SegmentType::PageFooter => {
                if ignore_headers_and_footers {
                    continue;
                }
                finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                current_segments.push(segment.clone());
                finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                current_word_count = 0;
                continue;
            }
            _ => {
                prev_hierarchy_level = current_hierarchy_level;
                if let Some((caption_idx, caption_word_count)) =
                    should_pair_with_caption(&segments, i, segment)
                {
                    let pair_length = segment_word_count + caption_word_count;

                    if pair_length >= target_length {
                        // If the current chunk has content, finalize it
                        if !current_segments.is_empty() {
                            finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                        }

                        // Add each item as its own chunk
                        if caption_idx < i {
                            chunks.push(Chunk::new(vec![segments[caption_idx].clone()]));
                            chunks.push(Chunk::new(vec![segment.clone()]));
                        } else {
                            chunks.push(Chunk::new(vec![segment.clone()]));
                            chunks.push(Chunk::new(vec![segments[caption_idx].clone()]));
                        }

                        current_word_count = 0;
                    } else {
                        if current_word_count + pair_length > target_length {
                            finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                            current_word_count = 0;
                        }

                        // Add both segments in the correct order
                        if caption_idx < i {
                            current_segments.push(segments[caption_idx].clone());
                            current_segments.push(segment.clone());
                            current_word_count += pair_length;
                        } else {
                            current_segments.push(segment.clone());
                            current_segments.push(segments[caption_idx].clone());
                            current_word_count += pair_length;
                        }
                    }
                    skip_indices.insert(caption_idx);
                } else if current_word_count + segment_word_count > target_length {
                    finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                    current_segments.push(segment.clone());
                    current_word_count = segment_word_count;
                } else {
                    current_segments.push(segment.clone());
                    current_word_count += segment_word_count;
                }
            }
        }
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
            html: None,
            image: None,
            llm: None,
            markdown: None,
            ocr: vec![],
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
}
