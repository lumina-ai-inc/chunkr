use crate::models::cropping::{CroppingStrategy, PictureCroppingStrategy};
use crate::models::output::{Segment, SegmentType};
use crate::models::pipeline::Pipeline;
use crate::models::segment_processing::GenerationStrategy;
use crate::models::task::Configuration;
use crate::utils::services::images::crop_image;
use rayon::prelude::*;
use std::error::Error;
use std::sync::Arc;
use tempfile::NamedTempFile;

trait CroppingBehavior {
    fn should_crop_all(&self) -> bool;
    fn should_crop_auto(&self) -> bool;
}

impl CroppingBehavior for CroppingStrategy {
    fn should_crop_all(&self) -> bool {
        matches!(self, CroppingStrategy::All)
    }

    fn should_crop_auto(&self) -> bool {
        matches!(self, CroppingStrategy::Auto)
    }
}

impl CroppingBehavior for PictureCroppingStrategy {
    fn should_crop_all(&self) -> bool {
        matches!(self, PictureCroppingStrategy::All)
    }

    fn should_crop_auto(&self) -> bool {
        matches!(self, PictureCroppingStrategy::Auto)
    }
}

fn should_crop<T: CroppingBehavior>(
    cropping_strategy: &T,
    strategy: &GenerationStrategy,
    llm: &Option<String>,
) -> bool {
    if cropping_strategy.should_crop_all() {
        true
    } else if cropping_strategy.should_crop_auto() {
        *strategy == GenerationStrategy::LLM || llm.is_some()
    } else {
        false
    }
}

async fn crop_segment(
    page_image: &NamedTempFile,
    configuration: &Configuration,
    segment: &Segment,
) -> Result<Option<NamedTempFile>, Box<dyn Error>> {
    let should_crop = match segment.segment_type {
        SegmentType::Formula | SegmentType::Page => {
            let config = match segment.segment_type {
                SegmentType::Formula => &configuration.segment_processing.formula,
                SegmentType::Page => &configuration.segment_processing.page,
                _ => unreachable!(),
            };
            match config {
                Some(config) => should_crop(&config.crop_image, &config.strategy, &config.llm),
                None => false,
            }
        }
        SegmentType::Table => {
            let config = &configuration.segment_processing.table;
            match config {
                Some(config) => should_crop(&config.crop_image, &config.strategy, &config.llm),
                None => false,
            }
        }
        SegmentType::Picture => {
            let config = &configuration.segment_processing.picture;
            match config {
                Some(config) => should_crop(&config.crop_image, &config.strategy, &config.llm),
                None => false,
            }
        }
        _ => {
            let config = match segment.segment_type {
                SegmentType::Title => &configuration.segment_processing.title,
                SegmentType::SectionHeader => &configuration.segment_processing.section_header,
                SegmentType::Text => &configuration.segment_processing.text,
                SegmentType::ListItem => &configuration.segment_processing.list_item,
                SegmentType::Caption => &configuration.segment_processing.caption,
                SegmentType::Footnote => &configuration.segment_processing.footnote,
                SegmentType::PageHeader => &configuration.segment_processing.page_header,
                SegmentType::PageFooter => &configuration.segment_processing.page_footer,
                _ => unreachable!(),
            };
            match config {
                Some(config) => should_crop(&config.crop_image, &config.strategy, &config.llm),
                None => false,
            }
        }
    };

    if should_crop {
        let cropped_image = crop_image(page_image, &segment.bbox)?;
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
    let page_images = pipeline.page_images.as_ref().unwrap();
    let configuration = pipeline.get_task()?.configuration.clone();
    let segment_images = pipeline.segment_images.clone();
    pipeline.chunks.par_iter().for_each(|chunk| {
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
    pipeline.segment_images = segment_images;
    Ok(())
}
