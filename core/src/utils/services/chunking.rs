use crate::models::{
    output::{Chunk, Segment, SegmentType},
    task::Configuration,
};
use rayon::prelude::*;

fn get_hierarchy_level(segment_type: &SegmentType) -> u32 {
    match segment_type {
        SegmentType::Title => 3,
        SegmentType::SectionHeader => 2,
        _ => 1,
    }
}

pub fn hierarchical_chunking(
    segments: Vec<Segment>,
    configuration: &Configuration,
) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut current_segments: Vec<Segment> = Vec::new();
    let mut current_word_count = 0;
    let target_length = configuration.chunk_processing.target_length;
    let ignore_headers_and_footers = configuration.chunk_processing.ignore_headers_and_footers;

    fn finalize_and_start_new_chunk(chunks: &mut Vec<Chunk>, segments: &mut Vec<Segment>) {
        if !segments.is_empty() {
            chunks.push(Chunk::new(segments.clone()));
            segments.clear();
        }
    }

    let mut prev_hierarchy_level = 1;
    let mut segment_paired = false;

    // Makes the chunking faster by calculating the word count in parallel
    segments.par_iter().for_each(|segment| {
        if let Err(e) = segment.count_embed_words(configuration) {
            println!("Error: {}", e);
        }
    });

    for (i, segment) in segments.iter().enumerate() {
        let segment_word_count = segment.count_embed_words(configuration)?;
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

                if (segment.segment_type == SegmentType::Picture
                    || segment.segment_type == SegmentType::Table)
                    && !segment_paired
                {
                    let next_is_caption = segments
                        .get(i + 1)
                        .is_some_and(|s| s.segment_type == SegmentType::Caption);
                    let caption_word_count = if let Some(s) = segments.get(i + 1) {
                        s.count_embed_words(configuration)?
                    } else {
                        0
                    };
                    if next_is_caption
                        && current_word_count + segment_word_count + caption_word_count
                            > target_length
                    {
                        finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                        current_segments.push(segment.clone());
                        current_word_count = segment_word_count;
                        default_chunk_behavior = false;
                        segment_paired = true; // Caption is paired with picture or table
                    }
                }
                if segment.segment_type == SegmentType::Caption && !segment_paired {
                    let next_is_asset = segments.get(i + 1).is_some_and(|s| {
                        s.segment_type == SegmentType::Picture
                            || s.segment_type == SegmentType::Table
                    });
                    let asset_word_count = if let Some(s) = segments.get(i + 1) {
                        s.count_embed_words(configuration)?
                    } else {
                        0
                    };
                    if next_is_asset
                        && current_word_count + segment_word_count + asset_word_count
                            > target_length
                    {
                        finalize_and_start_new_chunk(&mut chunks, &mut current_segments);
                        current_segments.push(segment.clone());
                        current_word_count = segment_word_count;
                        default_chunk_behavior = false;
                        segment_paired = true; // Picture or table is paired with caption
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
    use crate::models::chunk_processing::{ChunkProcessing, Tokenizer, TokenizerType};
    use crate::models::llm::LlmProcessing;
    use crate::models::output::{BoundingBox, Segment, SegmentType};
    use crate::models::segment_processing::{EmbedSource, SegmentProcessing};
    use crate::models::upload::{ErrorHandlingStrategy, OcrStrategy, SegmentationStrategy};

    fn create_segment(text: &str, segment_type: SegmentType) -> Segment {
        Segment {
            bbox: BoundingBox::new(0.0, 0.0, 0.0, 0.0),
            content: String::new(),
            confidence: None,
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
            text: text.to_string(),
        }
    }

    fn create_test_config(target_length: u32, ignore_headers_and_footers: bool) -> Configuration {
        let mut config = Configuration {
            chunk_processing: ChunkProcessing {
                ignore_headers_and_footers,
                target_length,
                tokenizer: TokenizerType::Enum(Tokenizer::Cl100kBase),
            },
            expires_in: None,
            high_resolution: false,
            input_file_url: None,
            json_schema: None,
            model: None,
            ocr_strategy: OcrStrategy::All,
            segment_processing: SegmentProcessing::default(),
            segmentation_strategy: SegmentationStrategy::LayoutAnalysis,
            target_chunk_length: None,
            error_handling: ErrorHandlingStrategy::default(),
            llm_processing: LlmProcessing::default(),
        };

        config
            .segment_processing
            .table
            .as_mut()
            .unwrap()
            .embed_sources = vec![EmbedSource::HTML, EmbedSource::Markdown];

        config
    }

    #[test]
    fn test_caption_picture_pairing() {
        let segments = vec![
            create_segment("Caption 1", SegmentType::Caption),
            create_segment("Picture 1", SegmentType::Picture),
            create_segment("Caption 2", SegmentType::Caption),
            create_segment("Picture 2", SegmentType::Picture),
        ];

        let chunks = hierarchical_chunking(segments, &create_test_config(100, true)).unwrap();

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

        let chunks = hierarchical_chunking(segments, &create_test_config(100, true)).unwrap();

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

        let chunks = hierarchical_chunking(segments, &create_test_config(15, true)).unwrap();

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

        let chunks = hierarchical_chunking(segments, &create_test_config(100, true)).unwrap();

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

        let chunks = hierarchical_chunking(segments, &create_test_config(100, true)).unwrap();

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

        let chunks = hierarchical_chunking(segments, &create_test_config(100, true)).unwrap();

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

        let chunks = hierarchical_chunking(segments, &create_test_config(10, true)).unwrap();

        println!("\n=== Chunk Contents ===");
        for (i, chunk) in chunks.iter().enumerate() {
            println!("\nChunk {}:", i);
            println!("{:?}", chunk.chunk_length);
            for segment in &chunk.segments {
                println!(
                    "  Type: {:?}, Content: \"{}\"",
                    segment.segment_type, segment.text
                );
            }
        }

        let caption_count = chunks
            .iter()
            .filter(|chunk| {
                chunk
                    .segments
                    .iter()
                    .any(|s| s.segment_type == SegmentType::Caption)
            })
            .count();

        println!("\nTotal chunks with captions: {}", caption_count);

        assert_eq!(
            caption_count, 1,
            "Caption should appear exactly once in the chunks"
        );
    }

    #[test]
    fn test_complex_pairing_sequences() {
        let test_cases = [
            (
                // Case 1: Multiple sequential pairs
                vec![
                    create_segment("Caption 1", SegmentType::Caption),
                    create_segment("Picture 1", SegmentType::Picture),
                    create_segment("Caption 2", SegmentType::Caption),
                    create_segment("Picture 2", SegmentType::Picture),
                ],
                vec![(0, 1), (2, 3)], // Expected pairs (indices)
            ),
            (
                // Case 2: Picture looking ahead for caption
                vec![
                    create_segment("Text start", SegmentType::Text),
                    create_segment("Picture 1", SegmentType::Picture),
                    create_segment("Caption 1", SegmentType::Caption),
                    create_segment("Text end", SegmentType::Text),
                ],
                vec![(1, 2)], // Picture 1 + Caption 1
            ),
            (
                // Case 3: Mixed asset types competing for caption
                vec![
                    create_segment("Picture 1", SegmentType::Picture),
                    create_segment("Table 1", SegmentType::Table),
                    create_segment("Caption 1", SegmentType::Caption),
                    create_segment("Text", SegmentType::Text),
                ],
                vec![(1, 2)], // Table 1 + Caption 1
            ),
            (
                // Case 4: Multiple captions and pictures interleaved
                vec![
                    create_segment("Caption 1", SegmentType::Caption),
                    create_segment("Caption 2", SegmentType::Caption),
                    create_segment("Picture 1", SegmentType::Picture),
                    create_segment("Picture 2", SegmentType::Picture),
                ],
                vec![(1, 2)], // Caption 2 + Picture 1
            ),
        ];

        for (case_index, (segments, expected_pairs)) in test_cases.iter().enumerate() {
            println!("\nTesting case {}", case_index + 1);
            let chunks =
                hierarchical_chunking(segments.clone(), &create_test_config(100, true)).unwrap();

            // Verify pairs stay together in the same chunk
            for &(first, second) in expected_pairs {
                let pair_chunk = chunks.iter().find(|chunk| {
                    chunk.segments.windows(2).any(|window| {
                        window[0].segment_type == segments[first].segment_type
                            && window[1].segment_type == segments[second].segment_type
                    })
                });
                assert!(
                    pair_chunk.is_some(),
                    "Case {}: Expected pair {:?} + {:?} not found together in any chunk",
                    case_index + 1,
                    segments[first].segment_type,
                    segments[second].segment_type
                );
            }
        }
    }

    #[test]
    fn test_chunk_size_boundaries() {
        let segments = vec![
            create_segment("Long text ".repeat(10).as_str(), SegmentType::Text),
            create_segment("Picture 1", SegmentType::Picture),
            create_segment("Short caption", SegmentType::Caption),
            create_segment("More long text ".repeat(10).as_str(), SegmentType::Text),
        ];

        let chunks = hierarchical_chunking(segments, &create_test_config(20, true)).unwrap();

        // Verify that Picture + Caption stay together even when chunks split
        assert!(
            chunks.len() > 1,
            "Should split into multiple chunks due to size"
        );

        // Find the chunk containing the picture
        let picture_chunk = chunks
            .iter()
            .find(|chunk| {
                chunk
                    .segments
                    .iter()
                    .any(|s| s.segment_type == SegmentType::Picture)
            })
            .unwrap();

        // Verify picture and caption are together
        let picture_index = picture_chunk
            .segments
            .iter()
            .position(|s| s.segment_type == SegmentType::Picture)
            .unwrap();
        assert_eq!(
            picture_chunk.segments[picture_index + 1].segment_type,
            SegmentType::Caption,
            "Caption should follow Picture in the same chunk"
        );
    }

    #[test]
    fn test_edge_cases() {
        let test_cases = [
            (
                // Case 1: Empty document
                vec![],
                0, // Expected chunks
            ),
            (
                // Case 2: Single segment
                vec![create_segment("Lonely caption", SegmentType::Caption)],
                1,
            ),
            (
                // Case 3: All captions
                vec![
                    create_segment("Caption 1", SegmentType::Caption),
                    create_segment("Caption 2", SegmentType::Caption),
                    create_segment("Caption 3", SegmentType::Caption),
                ],
                1,
            ),
            (
                // Case 4: Alternating types
                vec![
                    create_segment("Picture 1", SegmentType::Picture),
                    create_segment("Table 1", SegmentType::Table),
                    create_segment("Caption 1", SegmentType::Caption),
                    create_segment("Picture 2", SegmentType::Picture),
                ],
                1,
            ),
        ];

        for (case_index, (segments, expected_chunk_count)) in test_cases.iter().enumerate() {
            let chunks =
                hierarchical_chunking(segments.clone(), &create_test_config(1000, true)).unwrap();
            assert_eq!(
                chunks.len(),
                *expected_chunk_count,
                "Case {} failed: expected {} chunks, got {}",
                case_index + 1,
                expected_chunk_count,
                chunks.len()
            );
        }
    }

    #[test]
    fn test_hierarchy_with_pairs() {
        let segments = vec![
            create_segment("Title", SegmentType::Title),
            create_segment("Picture 1", SegmentType::Picture),
            create_segment("Caption 1", SegmentType::Caption),
            create_segment("Section 1", SegmentType::SectionHeader),
            create_segment("Table 1", SegmentType::Table),
            create_segment("Caption 2", SegmentType::Caption),
        ];

        let chunks = hierarchical_chunking(segments, &create_test_config(1000, true)).unwrap();

        // Verify that hierarchy changes create new chunks but pairs stay together
        assert!(chunks.len() > 1, "Should split on hierarchy changes");

        // Verify pairs in their respective hierarchical sections
        for chunk in &chunks {
            if chunk.segments[0].segment_type == SegmentType::Title {
                assert!(chunk.segments.windows(2).any(|w| (w[0].segment_type
                    == SegmentType::Picture
                    && w[1].segment_type == SegmentType::Caption)
                    || (w[0].segment_type == SegmentType::Caption
                        && w[1].segment_type == SegmentType::Picture)));
            }
            if chunk.segments[0].segment_type == SegmentType::SectionHeader {
                assert!(chunk.segments.windows(2).any(|w| (w[0].segment_type
                    == SegmentType::Table
                    && w[1].segment_type == SegmentType::Caption)
                    || (w[0].segment_type == SegmentType::Caption
                        && w[1].segment_type == SegmentType::Table)));
            }
        }
    }

    #[test]
    fn test_mixed_content_sequences() {
        let segments = vec![
            create_segment("Regular text 1", SegmentType::Text),
            create_segment("Picture 1", SegmentType::Picture),
            create_segment("Caption 1", SegmentType::Caption),
            create_segment("Regular text 2", SegmentType::Text),
            create_segment("Table 1", SegmentType::Table),
            create_segment("Regular text 3", SegmentType::Text),
            create_segment("Caption 2", SegmentType::Caption),
            create_segment("Picture 2", SegmentType::Picture),
        ];

        let chunks = hierarchical_chunking(segments, &create_test_config(30, true)).unwrap();

        // Print chunk contents for debugging
        for (i, chunk) in chunks.iter().enumerate() {
            println!("\nChunk {}:", i);
            for segment in &chunk.segments {
                println!("  {:?}: {}", segment.segment_type, segment.text);
            }
        }

        // Verify that pairs are maintained even with mixed content
        for chunk in &chunks {
            // Check for Picture+Caption pairs
            if chunk
                .segments
                .iter()
                .any(|s| s.segment_type == SegmentType::Picture)
            {
                assert!(chunk.segments.windows(2).any(|w| (w[0].segment_type
                    == SegmentType::Picture
                    && w[1].segment_type == SegmentType::Caption)
                    || (w[0].segment_type == SegmentType::Caption
                        && w[1].segment_type == SegmentType::Picture)));
            }
        }
    }
}
