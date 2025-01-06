use crate::models::chunkr::cropping::CroppingStrategy;
use crate::models::chunkr::output::{Segment, SegmentType};
use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::segment_processing::GenerationStrategy;
use crate::models::chunkr::task::{Configuration, Status};
use crate::utils::services::images::crop_image;
use rayon::prelude::*;
use std::error::Error;
use std::sync::Arc;
use tempfile::NamedTempFile;

async fn crop_segment(
    page_image: &NamedTempFile,
    configuration: &Configuration,
    segment: &Segment,
) -> Result<Option<NamedTempFile>, Box<dyn Error>> {
    let should_crop = match segment.segment_type {
        SegmentType::Table | SegmentType::Formula => {
            let config = match segment.segment_type {
                SegmentType::Table => &configuration.segment_processing.table,
                SegmentType::Formula => &configuration.segment_processing.formula,
                _ => unreachable!(),
            };
            match config.crop_image {
                CroppingStrategy::All => true,
                CroppingStrategy::Auto => {
                    config.html == GenerationStrategy::LLM
                        || config.markdown == GenerationStrategy::LLM
                        || config.llm.is_some()
                }
            }
        }
        _ => {
            let config = match segment.segment_type {
                SegmentType::Title => &configuration.segment_processing.title,
                SegmentType::SectionHeader => &configuration.segment_processing.section_header,
                SegmentType::Text => &configuration.segment_processing.text,
                SegmentType::ListItem => &configuration.segment_processing.list_item,
                SegmentType::Picture => &configuration.segment_processing.picture,
                SegmentType::Caption => &configuration.segment_processing.caption,
                SegmentType::Footnote => &configuration.segment_processing.footnote,
                SegmentType::PageHeader => &configuration.segment_processing.page_header,
                SegmentType::PageFooter => &configuration.segment_processing.page_footer,
                SegmentType::Page => &configuration.segment_processing.page,
                _ => unreachable!(),
            };
            match config.crop_image {
                CroppingStrategy::All => true,
                CroppingStrategy::Auto => {
                    config.html == GenerationStrategy::LLM
                        || config.markdown == GenerationStrategy::LLM
                        || config.llm.is_some()
                }
            }
        }
    };

    if should_crop {
        let cropped_image = crop_image(page_image, &segment.bbox)?;
        println!(
            "Croping image for segment type: {:?} and id: {:?}",
            segment.segment_type, segment.segment_id
        );
        Ok(Some(cropped_image))
    } else {
        Ok(None)
    }
}

/// Crop the segments
///
/// This function will crop the segments in parallel.
/// It will use the configuration to determine if cropping is enabled or required for downstream processing.
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn Error>> {
    pipeline
        .update_status(Status::Processing, Some("Cropping segments".to_string()))
        .await?;

    let page_images = pipeline.page_images.as_ref().unwrap();
    let configuration = pipeline
        .task_payload
        .as_ref()
        .unwrap()
        .current_configuration
        .clone();
    let segment_images = pipeline.segment_images.clone();

    pipeline.output.chunks.par_iter().for_each(|chunk| {
        chunk.segments.par_iter().for_each(|segment| {
            let page_image = page_images
                .get(segment.page_number as usize - 1)
                .unwrap()
                .as_ref();

            let cropped_image =
                futures::executor::block_on(crop_segment(page_image, &configuration, segment))
                    .expect("Failed to crop segment");

            if let Some(cropped_image) = cropped_image {
                segment_images.insert(segment.segment_id.clone(), Arc::new(cropped_image));
            }
        });
    });

    Ok(())
}